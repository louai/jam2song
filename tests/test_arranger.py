import numpy as np
import pytest

from jam2song.models import (
    AudioInfo, RenderParams, Segment, SectionSpec, Structure,
)
from jam2song.arranger import arrange, _energy_fit, _filter_candidates


def _make_segment(index, energy_tier="mid", trend="steady",
                  mean_energy=0.5, mean_brightness=0.5,
                  onset_density=0.5, duration=30.0,
                  source_start=0.0) -> Segment:
    return Segment(
        index=index,
        source_start=source_start,
        source_end=source_start + duration,
        duration=duration,
        mean_energy=mean_energy,
        mean_brightness=mean_brightness,
        onset_density=onset_density,
        energy_slope=0.0,
        internal_variance=0.01,
        energy_tier=energy_tier,
        trend=trend,
    )


def _make_dist_matrix(segments: list[Segment]) -> np.ndarray:
    n = len(segments)
    feats = np.array([[s.mean_energy, s.mean_brightness, s.onset_density] for s in segments])
    from scipy.spatial.distance import cdist
    return cdist(feats, feats, metric="euclidean")


def _make_audio_info(duration=600.0) -> AudioInfo:
    return AudioInfo(path="fake.wav", duration=duration,
                     sample_rate=44100, channels=2, tempo_bpm=120.0)


def _make_render_params() -> RenderParams:
    return RenderParams(crossfade=2.0, fade_in=3.0, fade_out=5.0, output_path="out.wav")


# --- filter_candidates ---

def test_filter_low_energy():
    segs = [
        _make_segment(0, energy_tier="low"),
        _make_segment(1, energy_tier="high"),
    ]
    spec = SectionSpec(role="intro", energy="low", relative_duration=1)
    result = _filter_candidates(segs, spec, used=set())
    assert len(result) == 1
    assert result[0].index == 0


def test_filter_excludes_used():
    segs = [
        _make_segment(0, energy_tier="low"),
        _make_segment(1, energy_tier="low"),
    ]
    spec = SectionSpec(role="intro", energy="low", relative_duration=1)
    result = _filter_candidates(segs, spec, used={0})
    assert len(result) == 1
    assert result[0].index == 1


def test_filter_rising_trend():
    segs = [
        _make_segment(0, energy_tier="mid", trend="rising"),
        _make_segment(1, energy_tier="mid", trend="steady"),
    ]
    spec = SectionSpec(role="build", energy="rising", relative_duration=1.5)
    result = _filter_candidates(segs, spec, used=set())
    assert len(result) == 1
    assert result[0].trend == "rising"


# --- energy_fit ---

def test_energy_fit_low_perfect():
    seg = _make_segment(0, energy_tier="low", trend="steady")
    assert _energy_fit(seg, "low") == pytest.approx(1.1)


def test_energy_fit_high_for_low():
    seg = _make_segment(0, energy_tier="high", trend="steady")
    assert _energy_fit(seg, "low") == pytest.approx(0.0)


def test_energy_fit_rising():
    seg = _make_segment(0, energy_tier="high", trend="rising")
    assert _energy_fit(seg, "rising") == pytest.approx(1.0)


def test_energy_fit_falling():
    seg = _make_segment(0, energy_tier="low", trend="falling")
    assert _energy_fit(seg, "falling") == pytest.approx(1.0)


# --- arrange ---

def _simple_structure(roles_energies):
    sections = [
        SectionSpec(role=r, energy=e, relative_duration=1.0)
        for r, e in roles_energies
    ]
    return Structure(name="test", description="", sections=sections)


def test_arrange_section_count_matches_structure():
    segs = [
        _make_segment(0, energy_tier="low"),
        _make_segment(1, energy_tier="mid"),
        _make_segment(2, energy_tier="high"),
    ]
    dist = _make_dist_matrix(segs)
    structure = _simple_structure([("intro", "low"), ("middle", "mid"), ("end", "high")])
    plan = arrange(segs, dist, structure, _make_audio_info(), 90.0, _make_render_params())
    assert len(plan.arranged_sections) == 3


def test_arrange_no_segment_reuse():
    segs = [
        _make_segment(0, energy_tier="low", mean_energy=0.1),
        _make_segment(1, energy_tier="mid", mean_energy=0.5),
        _make_segment(2, energy_tier="high", mean_energy=0.9),
    ]
    dist = _make_dist_matrix(segs)
    structure = _simple_structure([("a", "low"), ("b", "mid"), ("c", "high")])
    plan = arrange(segs, dist, structure, _make_audio_info(), 90.0, _make_render_params())
    indices = [a.segment.index for a in plan.arranged_sections]
    assert len(indices) == len(set(indices))


def test_arrange_selects_low_energy_for_low_role():
    segs = [
        _make_segment(0, energy_tier="high", mean_energy=0.9),
        _make_segment(1, energy_tier="low", mean_energy=0.1),
    ]
    dist = _make_dist_matrix(segs)
    structure = _simple_structure([("intro", "low")])
    plan = arrange(segs, dist, structure, _make_audio_info(), 30.0, _make_render_params())
    assert plan.arranged_sections[0].segment.energy_tier == "low"


def test_arrange_duration_scaling():
    """Sum of target_durations should approximate effective_target."""
    segs = [_make_segment(i, energy_tier="mid") for i in range(5)]
    dist = _make_dist_matrix(segs)
    structure = _simple_structure([(f"s{i}", "mid") for i in range(5)])
    target = 150.0
    crossfade = 2.0
    rp = RenderParams(crossfade=crossfade, fade_in=3.0, fade_out=5.0, output_path="out.wav")
    plan = arrange(segs, dist, structure, _make_audio_info(), target, rp)
    effective = target + crossfade * (5 - 1)
    total = sum(a.target_duration for a in plan.arranged_sections)
    assert total == pytest.approx(effective, rel=1e-6)


def test_arrange_similar_to_picks_closest():
    """With similar_to='first', second pick should be the segment closest to first."""
    # seg0: energy=0.9, brightness=0.9 (high)
    # seg1: energy=0.85, brightness=0.85 (high, very close to seg0)
    # seg2: energy=0.1, brightness=0.1  (high tier by construction, far from seg0)
    segs = [
        _make_segment(0, energy_tier="high", mean_energy=0.9, mean_brightness=0.9),
        _make_segment(1, energy_tier="high", mean_energy=0.85, mean_brightness=0.85),
        _make_segment(2, energy_tier="high", mean_energy=0.1, mean_brightness=0.1),
    ]
    dist = _make_dist_matrix(segs)
    sections = [
        SectionSpec(role="first", energy="high", relative_duration=1.0),
        SectionSpec(role="second", energy="high", relative_duration=1.0, similar_to="first"),
    ]
    structure = Structure(name="t", description="", sections=sections)
    plan = arrange(segs, dist, structure, _make_audio_info(), 60.0, _make_render_params())
    # first was seg0; second should be seg1 (closest to seg0), not seg2
    assert plan.arranged_sections[1].segment.index == 1


def test_arrange_positional_intro_prefers_early():
    """Intro should prefer a segment from the first 25% of the jam."""
    # Duration = 400s, first 25% = 0-100s
    segs = [
        _make_segment(0, energy_tier="low", source_start=10.0),   # early
        _make_segment(1, energy_tier="low", source_start=300.0),  # late
    ]
    dist = _make_dist_matrix(segs)
    structure = _simple_structure([("intro", "low")])
    plan = arrange(segs, dist, structure, _make_audio_info(duration=400.0), 30.0, _make_render_params())
    assert plan.arranged_sections[0].segment.source_start == pytest.approx(10.0)


def test_arrange_fallback_when_no_matching_energy():
    """If no segment matches the energy filter, fallback uses any unused segment."""
    segs = [_make_segment(0, energy_tier="mid")]  # only mid, but we ask for low
    dist = _make_dist_matrix(segs)
    structure = _simple_structure([("intro", "low")])
    plan = arrange(segs, dist, structure, _make_audio_info(), 30.0, _make_render_params())
    assert len(plan.arranged_sections) == 1


def test_arrange_score_range():
    """All scores should be in [0.0, 1.1] (base 0-1 + max 0.1 positional bonus)."""
    segs = [
        _make_segment(0, energy_tier="low", source_start=5.0),
        _make_segment(1, energy_tier="mid", source_start=50.0),
        _make_segment(2, energy_tier="high", source_start=150.0),
    ]
    dist = _make_dist_matrix(segs)
    structure = _simple_structure([("intro", "low"), ("mid", "mid"), ("outro", "high")])
    plan = arrange(segs, dist, structure, _make_audio_info(200.0), 90.0, _make_render_params())
    for arr in plan.arranged_sections:
        assert 0.0 <= arr.score <= 1.15  # slight margin for float precision


def test_arrange_actual_duration_capped_at_segment_length():
    """If segment is shorter than target, actual_duration = segment.duration."""
    short_seg = _make_segment(0, energy_tier="mid", duration=5.0)
    dist = _make_dist_matrix([short_seg])
    structure = _simple_structure([("only", "mid")])
    plan = arrange([short_seg], dist, structure, _make_audio_info(), 60.0, _make_render_params())
    assert plan.arranged_sections[0].actual_duration == pytest.approx(5.0)
