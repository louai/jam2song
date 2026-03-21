import numpy as np
import librosa
from scipy.signal import find_peaks

from .models import AnalysisResult, Segment


def segment(result: AnalysisResult, sensitivity: float = 0.5) -> list[Segment]:
    """Detect section boundaries via novelty curve and return a list of Segments."""
    novelty = np.asarray(result.novelty_curve)
    n_frames = len(novelty)
    beat_frames = np.asarray(result.beat_frames)

    # Map sensitivity [0.1, 1.0] to find_peaks parameters:
    # low sensitivity  → high prominence, large min distance (fewer, longer sections)
    # high sensitivity → low prominence, small min distance (more, shorter sections)
    prominence = 0.3 * (1.1 - sensitivity)
    frames_per_sec = result.sr_analysis / result.hop_length
    min_dist_sec = 1.0 + 3.0 * (1.0 - sensitivity)
    distance = max(1, int(min_dist_sec * frames_per_sec))

    peak_frames, _ = find_peaks(novelty, prominence=prominence, distance=distance)

    # Snap each peak to nearest beat
    snapped = _snap_to_beats(peak_frames, beat_frames)

    # Build boundary list: always start at 0, end at last frame
    boundaries = sorted(set([0] + snapped.tolist() + [n_frames - 1]))

    segments: list[Segment] = []
    for i in range(len(boundaries) - 1):
        start_frame = boundaries[i]
        end_frame = boundaries[i + 1]
        seg = _build_segment(len(segments), start_frame, end_frame, result)
        segments.append(seg)

    return segments


def _snap_to_beats(frames: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
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
