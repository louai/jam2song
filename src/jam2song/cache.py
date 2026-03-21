"""
Analysis cache — persist expensive analysis results to disk so subsequent
runs with different structures or target durations skip re-analysis.

Cache format:
  <source_stem>.cache.npz  — numpy arrays (y_analysis, y_render, features)
  <source_stem>.cache.json — scalar metadata + cache key

The cache is invalidated when the source file's mtime_ns or size changes,
or when SR_ANALYSIS changes. Stale or corrupt caches are silently discarded.
"""

import json
import os
from pathlib import Path

import numpy as np

from . import __version__
from .models import AnalysisResult, AudioInfo

CACHE_VERSION = 1


def _cache_paths(source_path: Path) -> tuple[Path, Path]:
    stem = source_path.parent / source_path.stem
    return Path(str(stem) + ".cache.npz"), Path(str(stem) + ".cache.json")


def _cache_key(source_path: Path, sr_analysis: int) -> dict:
    stat = source_path.stat()
    return {
        "cache_version": CACHE_VERSION,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
        "sr_analysis": sr_analysis,
        "jam2song_version": __version__,
    }


def load_cache(source_path: Path, sr_analysis: int) -> "AnalysisResult | None":
    """Return a cached AnalysisResult, or None if absent/stale/corrupt."""
    npz_path, json_path = _cache_paths(source_path)
    if not npz_path.exists() or not json_path.exists():
        return None
    try:
        stored = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    expected = _cache_key(source_path, sr_analysis)
    for field in ("cache_version", "source_mtime_ns", "source_size", "sr_analysis"):
        if stored.get(field) != expected[field]:
            return None

    try:
        data = np.load(npz_path, allow_pickle=False)
        meta = stored["audio_info"]
        audio_info = AudioInfo(
            path=meta["path"],
            duration=meta["duration"],
            sample_rate=meta["sample_rate"],
            channels=meta["channels"],
            tempo_bpm=meta["tempo_bpm"],
        )
        return AnalysisResult(
            audio_info=audio_info,
            y_analysis=data["y_analysis"],
            sr_analysis=int(stored["sr_analysis"]),
            y_render=data["y_render"],
            sr_render=int(meta["sr_render"]),
            hop_length=int(stored["hop_length"]),
            beat_frames=data["beat_frames"],
            beat_times=data["beat_times"],
            tempo_bpm=meta["tempo_bpm"],
            rms=data["rms"],
            spectral_centroid=data["spectral_centroid"],
            onset_strength=data["onset_strength"],
            novelty_curve=data["novelty_curve"],
        )
    except Exception:
        return None


def save_cache(source_path: Path, result: AnalysisResult) -> None:
    """Write cache files atomically. Silently ignores write errors."""
    npz_path, json_path = _cache_paths(source_path)
    key = _cache_key(source_path, result.sr_analysis)
    key["hop_length"] = result.hop_length
    key["audio_info"] = {
        "path": result.audio_info.path,
        "duration": result.audio_info.duration,
        "sample_rate": result.audio_info.sample_rate,
        "channels": result.audio_info.channels,
        "tempo_bpm": result.audio_info.tempo_bpm,
        "sr_render": result.sr_render,
    }

    tmp_json = json_path.with_suffix(".tmp")
    tmp_npz = Path(str(npz_path.with_suffix("")) + ".tmp")

    tmp_json.write_text(json.dumps(key, indent=2), encoding="utf-8")
    np.savez_compressed(
        str(tmp_npz),
        y_analysis=result.y_analysis,
        y_render=result.y_render,
        beat_frames=result.beat_frames,
        beat_times=result.beat_times,
        rms=result.rms,
        spectral_centroid=result.spectral_centroid,
        onset_strength=result.onset_strength,
        novelty_curve=result.novelty_curve,
    )
    # np.savez_compressed appends .npz — rename the actual file produced
    actual_npz = Path(str(tmp_npz) + ".npz")
    os.replace(actual_npz, npz_path)
    os.replace(tmp_json, json_path)
