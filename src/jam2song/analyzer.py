import os

import numpy as np
import librosa

from .models import AnalysisResult, AudioInfo


def _bootstrap_ffmpeg() -> None:
    """Add the imageio-ffmpeg bundled binary to PATH so librosa/audioread can find it."""
    try:
        import imageio_ffmpeg
        ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass  # fall back to system ffmpeg if anything goes wrong

HOP_LENGTH = 512
SR_ANALYSIS = 22050


def analyze(path: str, verbose: bool = False) -> AnalysisResult:
    _bootstrap_ffmpeg()
    # --- Phase 1: Load ---
    # Render copy: original quality, original channels
    y_render, sr_render = librosa.load(path, sr=None, mono=False)
    if y_render.ndim == 1:
        channels = 1
        duration = len(y_render) / sr_render
    else:
        channels = y_render.shape[0]
        duration = y_render.shape[1] / sr_render

    # Analysis copy: mono, 22050 Hz
    y_analysis, sr_analysis = librosa.load(path, sr=SR_ANALYSIS, mono=True)

    if verbose:
        mins = int(duration // 60)
        secs = duration % 60
        print(f"  Analysis copy: {len(y_analysis)} samples at {sr_analysis} Hz")
        print(f"  Render copy: {channels}ch at {sr_render} Hz, {mins}:{secs:05.2f}")

    # --- Phase 2: Feature Extraction ---
    tempo_arr, beat_frames = librosa.beat.beat_track(
        y=y_analysis, sr=sr_analysis, hop_length=HOP_LENGTH
    )
    tempo_bpm = float(np.squeeze(tempo_arr))
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr_analysis, hop_length=HOP_LENGTH
    )

    rms = librosa.feature.rms(y=y_analysis, hop_length=HOP_LENGTH)[0]
    spec_centroid = librosa.feature.spectral_centroid(
        y=y_analysis, sr=sr_analysis, hop_length=HOP_LENGTH
    )[0]
    onset_env = librosa.onset.onset_strength(
        y=y_analysis, sr=sr_analysis, hop_length=HOP_LENGTH
    )
    novelty = _compute_novelty(y_analysis)

    # Align all features to the same length (min of all)
    min_len = min(len(rms), len(spec_centroid), len(onset_env), len(novelty))
    rms = rms[:min_len]
    spec_centroid = spec_centroid[:min_len]
    onset_env = onset_env[:min_len]
    novelty = novelty[:min_len]

    audio_info = AudioInfo(
        path=path,
        duration=duration,
        sample_rate=sr_render,
        channels=channels,
        tempo_bpm=tempo_bpm,
    )

    return AnalysisResult(
        audio_info=audio_info,
        y_analysis=y_analysis,
        sr_analysis=sr_analysis,
        y_render=y_render,
        sr_render=sr_render,
        hop_length=HOP_LENGTH,
        beat_frames=beat_frames,
        beat_times=beat_times,
        tempo_bpm=tempo_bpm,
        rms=_normalize(rms),
        spectral_centroid=_normalize(spec_centroid),
        onset_strength=_normalize(onset_env),
        novelty_curve=_normalize(novelty),
    )


def _compute_novelty(y: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux as novelty curve."""
    S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    # Difference along time axis; prepend first frame to keep same length
    flux = np.diff(S, axis=1, prepend=S[:, :1])
    # Half-wave rectification: keep only positive changes
    flux = np.maximum(flux, 0.0)
    return flux.sum(axis=0)


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-10:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)
