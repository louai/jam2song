import numpy as np

from .models import (
    ArrangedSection,
    AudioInfo,
    RenderParams,
    ScoreBreakdown,
    Segment,
    SectionSpec,
    SongPlan,
    Structure,
)


def arrange(
    segments: list[Segment],
    dist_matrix: np.ndarray,
    structure: Structure,
    audio_info: AudioInfo,
    target_duration: float,
    render_params: RenderParams,
    beat_times: "np.ndarray | None" = None,
) -> SongPlan:
    n_sections = len(structure.sections)
    crossfade = render_params.crossfade

    # Crossfades overlap, so we need slightly more source material
    effective_target = target_duration + crossfade * (n_sections - 1)
    total_weight = sum(s.relative_duration for s in structure.sections)

    target_durations = {
        s.role: (s.relative_duration / total_weight) * effective_target
        for s in structure.sections
    }

    arranged: list[ArrangedSection] = []
    used_indices: set[int] = set()
    role_to_arranged: dict[str, ArrangedSection] = {}
    candidates_per_role: dict[str, list[tuple[int, float, ScoreBreakdown]]] = {}

    # Exclude near-silence segments from arrangement — these are typically dead
    # air at the start/end of a recording. 0.03 on the normalised 0-1 scale is
    # well below any musical content; genuine "low energy" sections sit at 0.1+.
    SILENCE_THRESHOLD = 0.03
    usable_segments = [s for s in segments if s.mean_energy > SILENCE_THRESHOLD]
    if not usable_segments:
        usable_segments = segments  # safety fallback

    # Pre-sort usable segments by position for intro/outro anchoring
    by_start = sorted(usable_segments, key=lambda s: s.source_start)
    by_end   = sorted(usable_segments, key=lambda s: s.source_end)

    # Temporal ordering: assign each non-anchor role an expected position fraction
    # (k+1)/(M+1) so roles spread evenly through the recording.
    non_anchor_roles = [
        s.role for s in structure.sections
        if not s.role.startswith(("intro", "outro"))
    ]
    M = len(non_anchor_roles)
    expected_fractions: dict[str, float] = {
        role: (k + 1) / (M + 1)
        for k, role in enumerate(non_anchor_roles)
    }
    # Shorter target duration → tighter temporal constraint.
    # temporal_weight 0.5 at 60 s, ~0.36 at 210 s, 0.0 at 600 s.
    temporal_weight = 0.5 * max(0.0, 1.0 - (target_duration - 60) / (600 - 60))

    prev_spec = None
    for spec in structure.sections:
        target_sec = target_durations[spec.role]

        # Intro and outro are anchored to the actual start/end of the recording.
        # They play at their natural length — no trimming to target_duration.
        if spec.role.startswith("intro"):
            # Sequential from source start — each intro* role takes the next
            # earliest unused segment, so intro → seg 0, intro_fill → seg 1, etc.
            candidates = [s for s in by_start if s.index not in used_indices][:1]
            if not candidates:
                candidates = [by_start[0]] if by_start else list(usable_segments[:1])
        elif spec.role.startswith("outro"):
            # Sequential from source end — each outro* role takes the next latest
            # unused segment working backwards.
            candidates = [s for s in reversed(by_end) if s.index not in used_indices][:1]
            if not candidates:
                candidates = [by_end[-1]] if by_end else list(usable_segments[-1:])
        else:
            candidates = _filter_candidates(usable_segments, spec, used_indices)
            if not candidates:
                candidates = [s for s in usable_segments if s.index not in used_indices]
            if not candidates:
                candidates = list(usable_segments)

        scored = _score_candidates(
            candidates, spec, target_sec, arranged, dist_matrix,
            role_to_arranged, audio_info,
            expected_fraction=expected_fractions.get(spec.role),
            temporal_weight=temporal_weight,
            prev_energy_spec=prev_spec.energy if prev_spec else None,
        )
        best_seg, score, breakdown = scored[0]
        candidates_per_role[spec.role] = [
            (seg.index, sc, bd) for seg, sc, bd in scored
        ]

        # Intro and outro use their natural length to preserve the real opening/
        # closing of the recording.  Cap at 2× the target so very long silence-
        # fade endings don't balloon the song duration.
        if spec.role.startswith(("intro", "outro")):
            cap = target_sec * 2.0
            raw = min(best_seg.duration, cap)
            actual_dur = _beat_snapped_duration(best_seg, raw, beat_times) if beat_times is not None else raw
        else:
            actual_dur = _beat_snapped_duration(best_seg, target_sec, beat_times)
        arr_sec = ArrangedSection(
            role=spec.role,
            segment=best_seg,
            target_duration=target_sec,
            actual_duration=actual_dur,
            score=score,
            score_breakdown=breakdown,
        )
        arranged.append(arr_sec)
        used_indices.add(best_seg.index)
        role_to_arranged[spec.role] = arr_sec
        prev_spec = spec

    return SongPlan(
        audio_info=audio_info,
        structure=structure,
        target_duration=target_duration,
        arranged_sections=arranged,
        render_params=render_params,
        candidates_per_role=candidates_per_role,
    )


def _filter_candidates(
    segments: list[Segment], spec: SectionSpec, used: set[int]
) -> list[Segment]:
    result = []
    for seg in segments:
        if seg.index in used:
            continue
        energy = spec.energy
        if energy == "low" and seg.energy_tier == "low":
            result.append(seg)
        elif energy == "mid" and seg.energy_tier == "mid":
            result.append(seg)
        elif energy == "high" and seg.energy_tier == "high":
            result.append(seg)
        elif energy == "rising" and seg.trend == "rising":
            result.append(seg)
        elif energy == "falling" and seg.trend == "falling":
            result.append(seg)
    return result


_ENERGY_LEVEL: dict[str, float] = {
    "low": 0.0, "mid": 1.0, "high": 2.0, "rising": 1.5, "falling": 0.5
}
_TRANSITION_WEIGHT = 0.15


def _score_candidates(
    candidates: list[Segment],
    spec: SectionSpec,
    target_sec: float,
    arranged: list[ArrangedSection],
    dist_matrix: np.ndarray,
    role_to_arranged: dict[str, ArrangedSection],
    audio_info: AudioInfo,
    expected_fraction: float | None = None,
    temporal_weight: float = 0.0,
    prev_energy_spec: str | None = None,
) -> list[tuple[Segment, float, ScoreBreakdown]]:
    jam_duration = audio_info.duration
    intro_cutoff = jam_duration * 0.25
    outro_start = jam_duration * 0.75
    max_dist = dist_matrix.max() if dist_matrix.size > 0 and dist_matrix.max() > 0 else 1.0

    scores = []
    for seg in candidates:
        energy_fit = _energy_fit(seg, spec.energy)

        # Variety vs. similarity
        if spec.similar_to is not None and spec.similar_to in role_to_arranged:
            ref_idx = role_to_arranged[spec.similar_to].segment.index
            raw_dist = float(dist_matrix[seg.index, ref_idx])
            variety_or_sim = 1.0 - (raw_dist / max_dist)
        elif arranged:
            used_idxs = [a.segment.index for a in arranged]
            dists = [float(dist_matrix[seg.index, idx]) for idx in used_idxs]
            variety_or_sim = float(np.mean(dists)) / max_dist
        else:
            variety_or_sim = 1.0

        # Duration fit: how close is natural duration to target
        duration_fit = min(seg.duration, target_sec) / max(seg.duration, target_sec)

        # Temporal fit: how close the segment is to its expected position in the source
        if expected_fraction is not None and temporal_weight > 0 and jam_duration > 0:
            segment_fraction = seg.source_start / jam_duration
            temporal_fit = 1.0 - abs(segment_fraction - expected_fraction)
        else:
            temporal_fit = 1.0

        # Transition score: does this candidate's energy move in the right direction
        # relative to the previously selected segment?
        if prev_energy_spec and arranged:
            prev_energy = arranged[-1].segment.mean_energy
            direction = (
                _ENERGY_LEVEL.get(spec.energy, 1.0)
                - _ENERGY_LEVEL.get(prev_energy_spec, 1.0)
            )
            diff = seg.mean_energy - prev_energy
            if direction > 0:    # template expects energy increase
                transition_score = float(0.5 + 0.5 * np.tanh(diff * 10))
            elif direction < 0:  # template expects energy decrease
                transition_score = float(0.5 - 0.5 * np.tanh(diff * 10))
            else:                # same level — neutral
                transition_score = 1.0
        else:
            transition_score = 1.0

        # Positional preference as a soft tiebreaker bonus
        positional = 0.0
        if spec.role.startswith("intro") and seg.source_start <= intro_cutoff:
            positional = 0.15
        elif spec.role.startswith("outro") and seg.source_start >= outro_start:
            positional = 0.20
        # Penalise using the very first segment as a breakdown/bridge — it's
        # often recording startup noise or a tentative opening, not a settled groove.
        if not spec.role.startswith("intro") and seg.source_start == 0.0:
            positional -= 0.15

        score = (
            energy_fit + variety_or_sim + duration_fit
            + temporal_weight * temporal_fit
            + _TRANSITION_WEIGHT * transition_score
        ) / (3.0 + temporal_weight + _TRANSITION_WEIGHT) + positional
        breakdown = ScoreBreakdown(
            energy_fit=energy_fit,
            variety=variety_or_sim,
            duration_fit=duration_fit,
            temporal_fit=temporal_fit,
            transition_score=transition_score,
        )
        scores.append((seg, score, breakdown))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _beat_snapped_duration(seg: Segment, target_sec: float, beat_times: "np.ndarray | None") -> float:
    """Return the actual duration to use, snapped to the nearest beat <= target."""
    raw = min(seg.duration, target_sec)
    if beat_times is None or raw >= seg.duration:
        return raw  # no trimming needed, or no beat info
    end_time = seg.source_start + raw
    # Beats that fall within this segment
    seg_beats = beat_times[(beat_times >= seg.source_start) & (beat_times <= seg.source_end)]
    if len(seg_beats) == 0:
        return raw
    # Find the beat closest to end_time but not past it
    before = seg_beats[seg_beats <= end_time]
    if len(before) == 0:
        return raw
    nearest = float(before[np.argmin(np.abs(before - end_time))])
    snapped = nearest - seg.source_start
    return snapped if snapped > 0 else raw


def _energy_fit(seg: Segment, energy_spec: str) -> float:
    tier = seg.energy_tier
    trend = seg.trend
    if energy_spec == "low":
        base = {"low": 1.0, "mid": 0.4, "high": 0.0}[tier]
        return base * (1.1 if trend == "steady" else 1.0)
    elif energy_spec == "mid":
        return {"low": 0.3, "mid": 1.0, "high": 0.3}[tier]
    elif energy_spec == "high":
        base = {"low": 0.0, "mid": 0.4, "high": 1.0}[tier]
        return base * (1.1 if trend == "steady" else 1.0)
    elif energy_spec == "rising":
        trend_score = {"rising": 1.0, "steady": 0.3, "falling": 0.0}[trend]
        tier_score = {"low": 0.2, "mid": 0.8, "high": 1.0}[tier]  # low tier is a poor build section
        return (trend_score + tier_score) / 2.0
    elif energy_spec == "falling":
        trend_score = {"falling": 1.0, "steady": 0.3, "rising": 0.0}[trend]
        tier_score = {"high": 0.5, "mid": 0.8, "low": 1.0}[tier]
        return (trend_score + tier_score) / 2.0
    return 0.5
