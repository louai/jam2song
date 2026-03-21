import numpy as np
import librosa
from scipy.signal import find_peaks

from .models import AnalysisResult, Segment

BEATS_PER_BAR = 4
BARS_PER_PHRASE = 2   # 2-bar / 8-beat phrases; standard unit for loop-based music


def segment(result: AnalysisResult, sensitivity: float = 0.5) -> list[Segment]:
    """Detect section boundaries at phrase-grid positions and return Segments.

    Boundary detection uses an enhanced novelty curve: spectral flux (novelty)
    plus a "fill bonus" — elevated onset activity in the 2 beats before a
    candidate boundary.  This makes the detector prefer cuts that follow drum
    fills.  All cut points are then snapped to the nearest 2-bar phrase
    boundary so that every edit lands on a consistent loop position.
    """
    novelty = np.asarray(result.novelty_curve)
    onset_env = np.asarray(result.onset_strength)
    beat_frames = np.asarray(result.beat_frames)
    n_frames = len(novelty)
    frames_per_sec = result.sr_analysis / result.hop_length

    # Estimate frames per beat from the actual beat grid
    if len(beat_frames) > 1:
        frames_per_beat = float(np.median(np.diff(beat_frames)))
    else:
        frames_per_beat = frames_per_sec * 60.0 / result.tempo_bpm

    beats_per_phrase = BEATS_PER_BAR * BARS_PER_PHRASE  # 8

    # -----------------------------------------------------------------
    # Fill bonus: causal moving average of onset_env over last 2 beats.
    # A drum fill raises onset activity → boosts the novelty score at
    # the phrase boundary where the fill ends.
    # -----------------------------------------------------------------
    lookback = max(1, int(2 * frames_per_beat))
    onset_limited = onset_env[:n_frames]
    cs = np.concatenate([[0.0], np.cumsum(onset_limited)])
    start_idx = np.maximum(0, np.arange(n_frames) - lookback)
    end_idx = np.arange(n_frames)
    counts = np.maximum(end_idx - start_idx, 1)
    fill_bonus = (cs[end_idx] - cs[start_idx]) / counts

    # Combined novelty: spectral flux + 30 % fill activity
    combined = novelty + 0.3 * fill_bonus
    cmax = combined.max()
    if cmax > 1e-10:
        combined = combined / cmax

    # -----------------------------------------------------------------
    # Peak detection — same sensitivity-to-params mapping as before
    # -----------------------------------------------------------------
    prominence = 0.05 + 0.20 * (1.0 - sensitivity)
    min_dist_sec = 8.0 + 22.0 * (1.0 - sensitivity)
    distance = max(1, int(min_dist_sec * frames_per_sec))

    peak_frames, _ = find_peaks(combined, prominence=prominence, distance=distance)

    # -----------------------------------------------------------------
    # Snap to nearest 2-bar phrase boundary
    # -----------------------------------------------------------------
    if len(beat_frames) >= beats_per_phrase:
        phrase_frames = beat_frames[::beats_per_phrase]
    else:
        phrase_frames = beat_frames

    snapped = _snap_to_phrases(peak_frames, phrase_frames)

    # Build boundary list: always start at 0, end at last frame
    boundaries = sorted(set([0] + list(snapped) + [n_frames - 1]))

    # Hard minimum: no segment shorter than 5 s (handles beat-snap collisions)
    min_seg_frames = int(5.0 * frames_per_sec)
    filtered = [boundaries[0]]
    for b in boundaries[1:]:
        if b - filtered[-1] >= min_seg_frames:
            filtered.append(b)
    boundaries = filtered

    segments: list[Segment] = []
    for i in range(len(boundaries) - 1):
        seg = _build_segment(len(segments), boundaries[i], boundaries[i + 1], result)
        segments.append(seg)
    return segments


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _snap_to_phrases(frames: np.ndarray, phrase_frames: np.ndarray) -> list[int]:
    """Snap each frame to the nearest 2-bar phrase boundary."""
    if len(phrase_frames) == 0 or len(frames) == 0:
        return frames.tolist() if hasattr(frames, 'tolist') else list(frames)
    result = []
    for f in frames:
        idx = int(np.argmin(np.abs(phrase_frames - f)))
        result.append(int(phrase_frames[idx]))
    return result


def _snap_to_beats(frames: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    """Snap frames to the nearest beat (kept for tests and fallback use)."""
    if len(beat_frames) == 0 or len(frames) == 0:
        return frames
    snapped = []
    for f in frames:
        idx = int(np.argmin(np.abs(beat_frames - f)))
        snapped.append(int(beat_frames[idx]))
    return np.array(snapped, dtype=int)


def _build_segment(index: int, start_frame: int, end_frame: int, result: AnalysisResult) -> Segment:
    sl = slice(start_frame, end_frame)
    rms_seg = np.asarray(result.rms)[sl]
    brightness_seg = np.asarray(result.spectral_centroid)[sl]
    onset_seg = np.asarray(result.onset_strength)[sl]

    source_start = librosa.frames_to_time(
        start_frame, sr=result.sr_analysis, hop_length=result.hop_length
    )
    source_end = librosa.frames_to_time(
        end_frame, sr=result.sr_analysis, hop_length=result.hop_length
    )
    duration = float(source_end - source_start)

    n = len(rms_seg)
    if n > 1:
        t = np.arange(n, dtype=float) / (n - 1)
        coeffs = np.polyfit(t, rms_seg, 1)
        energy_slope = float(coeffs[0])
    else:
        energy_slope = 0.0

    onset_density = float(onset_seg.sum()) / max(duration, 1e-6)

    return Segment(
        index=index,
        source_start=float(source_start),
        source_end=float(source_end),
        duration=duration,
        mean_energy=float(rms_seg.mean()) if n > 0 else 0.0,
        mean_brightness=float(brightness_seg.mean()) if n > 0 else 0.0,
        onset_density=onset_density,
        energy_slope=energy_slope,
        internal_variance=float(rms_seg.var()) if n > 0 else 0.0,
    )
