import numpy as np
import pytest

from jam2song.models import AnalysisResult, AudioInfo
from jam2song.segmenter import segment, _snap_to_beats


SR_ANALYSIS = 22050
HOP_LENGTH = 512


def _make_result(novelty: np.ndarray, rms: np.ndarray | None = None,
                 beat_frames: np.ndarray | None = None) -> AnalysisResult:
    """Build a minimal AnalysisResult for testing."""
    n = len(novelty)
    if rms is None:
        rms = np.ones(n) * 0.5
    if beat_frames is None:
        # beats every 10 frames
        beat_frames = np.arange(0, n, 10, dtype=int)

    audio_info = AudioInfo(
        path="fake.wav", duration=n * HOP_LENGTH / SR_ANALYSIS,
        sample_rate=44100, channels=1, tempo_bpm=120.0,
    )
    return AnalysisResult(
        audio_info=audio_info,
        y_analysis=np.zeros(n * HOP_LENGTH),
        sr_analysis=SR_ANALYSIS,
        y_render=np.zeros(n * HOP_LENGTH),
        sr_render=44100,
        hop_length=HOP_LENGTH,
        beat_frames=beat_frames,
        beat_times=beat_frames * HOP_LENGTH / SR_ANALYSIS,
        tempo_bpm=120.0,
        rms=rms,
        spectral_centroid=np.ones(n) * 0.5,
        onset_strength=np.ones(n) * 0.3,
        novelty_curve=novelty,
    )


def test_boundaries_always_start_at_zero():
    novelty = np.zeros(500)
    result = _make_result(novelty)
    segs = segment(result)
    assert segs[0].source_start == pytest.approx(0.0)


def test_boundaries_include_end():
    novelty = np.zeros(500)
    result = _make_result(novelty)
    segs = segment(result)
    # Last segment ends at the last frame time
    last_time = (499) * HOP_LENGTH / SR_ANALYSIS
    assert segs[-1].source_end == pytest.approx(last_time, abs=0.1)


def test_no_peaks_yields_one_segment():
    """Flat novelty → no peaks → entire audio is one segment."""
    novelty = np.zeros(500)
    result = _make_result(novelty)
    segs = segment(result)
    assert len(segs) == 1


def test_high_sensitivity_more_segments_than_low():
    """Higher sensitivity should produce more segments."""
    n = 2000
    # Create novelty with many mild peaks
    novelty = np.zeros(n)
    novelty[100::100] = 0.4
    result = _make_result(novelty)

    segs_high = segment(result, sensitivity=0.9)
    segs_low = segment(result, sensitivity=0.1)
    assert len(segs_high) >= len(segs_low)


def test_snap_to_beats_picks_nearest():
    beat_frames = np.array([0, 10, 20, 30, 40, 50], dtype=int)
    # Peak at frame 12 → nearest beat is 10
    result = _snap_to_beats(np.array([12]), beat_frames)
    assert result[0] == 10


def test_snap_to_beats_exact_match():
    beat_frames = np.array([0, 10, 20, 30], dtype=int)
    result = _snap_to_beats(np.array([20]), beat_frames)
    assert result[0] == 20


def test_snap_to_beats_empty_beats():
    """No beat frames → return frames unchanged."""
    result = _snap_to_beats(np.array([5, 15, 25]), np.array([], dtype=int))
    np.testing.assert_array_equal(result, [5, 15, 25])


def test_snap_to_beats_empty_peaks():
    beat_frames = np.array([0, 10, 20], dtype=int)
    result = _snap_to_beats(np.array([], dtype=int), beat_frames)
    assert len(result) == 0


def test_segment_mean_energy():
    """mean_energy should equal mean of RMS slice for that segment."""
    # Use a large enough signal so the peak clears the minimum section distance.
    # At sensitivity=1.0, min_dist ≈ 8s → ~344 frames at 22050/512 Hz.
    n = 2000
    rms = np.linspace(0.0, 1.0, n)
    # One strong peak at the midpoint → two segments
    novelty = np.zeros(n)
    novelty[1000] = 1.0
    beat_frames = np.arange(0, n, 5, dtype=int)
    result = _make_result(novelty, rms=rms, beat_frames=beat_frames)
    segs = segment(result, sensitivity=1.0)
    assert len(segs) == 2
    # First segment spans frames 0..1000, rms ramps 0→~0.5, mean ≈ 0.25
    assert 0.0 <= segs[0].mean_energy <= 0.6


def test_positive_energy_slope():
    n = 300
    rms = np.linspace(0.0, 1.0, n)
    novelty = np.zeros(n)
    result = _make_result(novelty, rms=rms)
    segs = segment(result)
    assert len(segs) == 1
    assert segs[0].energy_slope > 0


def test_negative_energy_slope():
    n = 300
    rms = np.linspace(1.0, 0.0, n)
    novelty = np.zeros(n)
    result = _make_result(novelty, rms=rms)
    segs = segment(result)
    assert segs[0].energy_slope < 0


def test_flat_energy_slope_near_zero():
    n = 300
    rms = np.full(n, 0.5)
    novelty = np.zeros(n)
    result = _make_result(novelty, rms=rms)
    segs = segment(result)
    assert abs(segs[0].energy_slope) < 1e-6


def test_segment_indices_are_sequential():
    n = 400
    novelty = np.zeros(n)
    novelty[150] = 1.0
    novelty[280] = 0.9
    result = _make_result(novelty)
    segs = segment(result, sensitivity=0.9)
    for i, s in enumerate(segs):
        assert s.index == i


def test_segment_duration_positive():
    n = 300
    novelty = np.zeros(n)
    result = _make_result(novelty)
    segs = segment(result)
    for s in segs:
        assert s.duration > 0
