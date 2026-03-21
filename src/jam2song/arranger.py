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

    for spec in structure.sections:
        target_sec = target_durations[spec.role]
        candidates = _filter_candidates(segments, spec, used_indices)

        if not candidates:
            # Fallback: relax energy constraint, use any unused segment
            candidates = [s for s in segments if s.index not in used_indices]
        if not candidates:
            # Last resort: allow reuse (more roles than segments)
            candidates = list(segments)

        scored = _score_candidates(
            candidates, spec, target_sec, arranged, dist_matrix,
            role_to_arranged, audio_info,
        )
        best_seg, score, breakdown = scored[0]

        actual_dur = min(best_seg.duration, target_sec)
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

    return SongPlan(
        audio_info=audio_info,
        structure=structure,
        target_duration=target_duration,
        arranged_sections=arranged,
        render_params=render_params,
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


def _score_candidates(
    candidates: list[Segment],
    spec: SectionSpec,
    target_sec: float,
    arranged: list[ArrangedSection],
    dist_matrix: np.ndarray,
    role_to_arranged: dict[str, ArrangedSection],
    audio_info: AudioInfo,
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

        # Positional preference as a soft tiebreaker bonus
        positional = 0.0
        if spec.role == "intro" and seg.source_start <= intro_cutoff:
            positional = 0.1
        elif spec.role == "outro" and seg.source_start >= outro_start:
            positional = 0.1

        score = (energy_fit + variety_or_sim + duration_fit) / 3.0 + positional
        breakdown = ScoreBreakdown(
            energy_fit=energy_fit,
            variety=variety_or_sim,
            duration_fit=duration_fit,
        )
        scores.append((seg, score, breakdown))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


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
        tier_score = {"low": 0.5, "mid": 0.8, "high": 1.0}[tier]
        return (trend_score + tier_score) / 2.0
    elif energy_spec == "falling":
        trend_score = {"falling": 1.0, "steady": 0.3, "rising": 0.0}[trend]
        tier_score = {"high": 0.5, "mid": 0.8, "low": 1.0}[tier]
        return (trend_score + tier_score) / 2.0
    return 0.5
