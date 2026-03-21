import numpy as np
import pytest

from jam2song.renderer import (
    _equal_power_crossfade,
    _apply_fade_in,
    _apply_fade_out,
    _join_with_crossfades,
)


# --- equal-power property ---

def test_equal_power_gain_curves():
    """cos²(t) + sin²(t) = 1 at every point in the crossfade."""
    n = 200
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    gain_out = np.cos(t * np.pi / 2)
    gain_in = np.sin(t * np.pi / 2)
    np.testing.assert_allclose(gain_out**2 + gain_in**2, np.ones(n), atol=1e-12)


def test_crossfade_output_length():
    """Output length = len(a) + len(b) - n."""
    a = np.ones(1000)
    b = np.ones(1000)
    n = 100
    result = _equal_power_crossfade(a, b, n)
    assert len(result) == 1000 + 1000 - n


def test_crossfade_zero_n_is_concatenation():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = _equal_power_crossfade(a, b, 0)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_crossfade_stereo_shape():
    """Stereo (2, N) shape is preserved."""
    a = np.ones((2, 1000))
    b = np.ones((2, 1000))
    n = 100
    result = _equal_power_crossfade(a, b, n)
    assert result.shape == (2, 1000 + 1000 - n)


def test_crossfade_short_segment_guard():
    """If a segment is shorter than n, actual_n is clamped."""
    a = np.ones(20)    # very short
    b = np.ones(1000)
    n = 100            # larger than half of a → clamped to 20//2 = 10
    result = _equal_power_crossfade(a, b, n)
    # Should not raise and output length should be 20 + 1000 - 10 = 1010
    assert len(result) == 1010


def test_crossfade_values_at_overlap_start():
    """At the start of the overlap, gain_out ≈ 1 and gain_in ≈ 0."""
    a = np.ones(1000)
    b = np.ones(1000) * 2.0
    n = 200
    result = _equal_power_crossfade(a, b, n)
    # At position len(a)-n (start of overlap), output ≈ 1*gain_out(0) + 2*gain_in(0) ≈ 1
    overlap_start = result[len(a) - n]
    assert overlap_start == pytest.approx(1.0, abs=0.01)


def test_crossfade_values_at_overlap_end():
    """At the end of the overlap, gain_out ≈ 0 and gain_in ≈ 1."""
    a = np.ones(1000)
    b = np.ones(1000) * 2.0
    n = 200
    result = _equal_power_crossfade(a, b, n)
    # At position len(a)-1 (end of overlap), output ≈ 1*0 + 2*1 = 2
    overlap_end = result[len(a) - 1]
    assert overlap_end == pytest.approx(2.0, abs=0.01)


# --- join_with_crossfades ---

def test_join_three_segments_length():
    a = np.ones(1000)
    b = np.ones(1000)
    c = np.ones(1000)
    n = 100
    result = _join_with_crossfades([a, b, c], n)
    # 1000 + 1000 - 100 + 1000 - 100 = 2800
    assert len(result) == 2800


def test_join_single_segment():
    a = np.ones(500)
    result = _join_with_crossfades([a], 50)
    np.testing.assert_array_equal(result, a)


def test_join_empty_raises():
    with pytest.raises(ValueError):
        _join_with_crossfades([], 50)


# --- fade_in ---

def test_fade_in_starts_at_zero():
    audio = np.ones(1000)
    result = _apply_fade_in(audio, 100)
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_fade_in_ends_at_original():
    audio = np.ones(1000)
    result = _apply_fade_in(audio, 100)
    # After the fade region, audio should be unchanged
    np.testing.assert_allclose(result[100:], 1.0)


def test_fade_in_monotone_increasing():
    audio = np.ones(1000)
    result = _apply_fade_in(audio, 200)
    fade_region = result[:200]
    diffs = np.diff(fade_region)
    assert np.all(diffs >= -1e-10)


def test_fade_in_stereo():
    audio = np.ones((2, 1000))
    result = _apply_fade_in(audio, 100)
    assert result.shape == (2, 1000)
    np.testing.assert_allclose(result[:, 0], 0.0, atol=1e-6)


# --- fade_out ---

def test_fade_out_ends_at_zero():
    audio = np.ones(1000)
    result = _apply_fade_out(audio, 100)
    assert result[-1] == pytest.approx(0.0, abs=1e-6)


def test_fade_out_starts_at_original():
    audio = np.ones(1000)
    result = _apply_fade_out(audio, 100)
    np.testing.assert_allclose(result[:-100], 1.0)


def test_fade_out_monotone_decreasing():
    audio = np.ones(1000)
    result = _apply_fade_out(audio, 200)
    fade_region = result[-200:]
    diffs = np.diff(fade_region)
    assert np.all(diffs <= 1e-10)


def test_fade_out_stereo():
    audio = np.ones((2, 1000))
    result = _apply_fade_out(audio, 100)
    assert result.shape == (2, 1000)
    np.testing.assert_allclose(result[:, -1], 0.0, atol=1e-6)


# --- edge cases ---

def test_fade_in_larger_than_audio():
    """Fade larger than audio: clamp to audio length."""
    audio = np.ones(50)
    result = _apply_fade_in(audio, 200)
    assert result[0] == pytest.approx(0.0, abs=1e-6)
    assert len(result) == 50


def test_fade_out_larger_than_audio():
    audio = np.ones(50)
    result = _apply_fade_out(audio, 200)
    assert result[-1] == pytest.approx(0.0, abs=1e-6)
    assert len(result) == 50
