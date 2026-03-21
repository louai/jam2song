import numpy as np
import soundfile as sf

from .models import AnalysisResult, SongPlan


def render(plan: SongPlan, analysis: AnalysisResult, output_path: str) -> float:
    """Render the song plan to an audio file. Returns the output duration in seconds."""
    sr = analysis.sr_render
    crossfade_samples = int(plan.render_params.crossfade * sr)
    fade_in_samples = int(plan.render_params.fade_in * sr)
    fade_out_samples = int(plan.render_params.fade_out * sr)

    y_render = np.asarray(analysis.y_render)

    # Extract each arranged section from the original quality audio
    segments_audio: list[np.ndarray] = []
    for arr_sec in plan.arranged_sections:
        seg = arr_sec.segment
        start_sample = int(seg.source_start * sr)
        end_sample = int((seg.source_start + arr_sec.actual_duration) * sr)
        # Clamp to valid range
        total_samples = y_render.shape[-1]
        start_sample = max(0, min(start_sample, total_samples - 1))
        end_sample = max(start_sample + 1, min(end_sample, total_samples))
        chunk = y_render[..., start_sample:end_sample]
        segments_audio.append(chunk)

    # Join with equal-power crossfades
    result = _join_with_crossfades(segments_audio, crossfade_samples)

    # Fade-in: if the song starts at the very beginning of the source recording
    # it already has a natural start — applying a long fade-in would bury the
    # drum fill or any content that kicks off the jam.  Use only a short
    # click-prevention fade (50 ms) in that case.
    first_source_start = plan.arranged_sections[0].segment.source_start
    if first_source_start < 0.5:
        fade_in_samples = min(fade_in_samples, int(0.05 * sr))

    result = _apply_fade_in(result, fade_in_samples)
    result = _apply_fade_out(result, fade_out_samples)

    # Write output
    if output_path.lower().endswith(".mp3"):
        _write_mp3(result, sr, output_path)
    else:
        # soundfile expects (samples, channels) for multi-channel
        if result.ndim == 2:
            sf.write(output_path, result.T, sr, subtype="PCM_24")
        else:
            sf.write(output_path, result, sr, subtype="PCM_24")

    return result.shape[-1] / sr


def _equal_power_crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Crossfade the last n samples of `a` with the first n samples of `b`
    using equal-power (cos/sin) curves. Returns the joined array.
    """
    if n <= 0:
        return np.concatenate([a, b], axis=-1)

    # Guard: n cannot exceed half the length of either array
    n = min(n, a.shape[-1] // 2, b.shape[-1] // 2)
    if n <= 0:
        return np.concatenate([a, b], axis=-1)

    t = np.linspace(0.0, 1.0, n, endpoint=False)
    gain_out = np.cos(t * np.pi / 2)
    gain_in = np.sin(t * np.pi / 2)

    if a.ndim == 2:
        gain_out = gain_out[np.newaxis, :]
        gain_in = gain_in[np.newaxis, :]

    overlap = a[..., -n:] * gain_out + b[..., :n] * gain_in
    return np.concatenate([a[..., :-n], overlap, b[..., n:]], axis=-1)


def _join_with_crossfades(segments: list[np.ndarray], n: int) -> np.ndarray:
    if not segments:
        raise ValueError("No segments to join")
    result = segments[0]
    for seg in segments[1:]:
        result = _equal_power_crossfade(result, seg, n)
    return result


def _apply_fade_in(audio: np.ndarray, n: int) -> np.ndarray:
    n = min(n, audio.shape[-1])
    if n <= 0:
        return audio
    t = np.linspace(0.0, 1.0, n)
    curve = np.sin(t * np.pi / 2)  # 0 → 1
    if audio.ndim == 2:
        curve = curve[np.newaxis, :]
    audio = audio.copy()
    audio[..., :n] *= curve
    return audio


def _apply_fade_out(audio: np.ndarray, n: int) -> np.ndarray:
    n = min(n, audio.shape[-1])
    if n <= 0:
        return audio
    t = np.linspace(0.0, 1.0, n)
    curve = np.cos(t * np.pi / 2)  # 1 → 0
    if audio.ndim == 2:
        curve = curve[np.newaxis, :]
    audio = audio.copy()
    audio[..., -n:] *= curve
    return audio


def _write_mp3(audio: np.ndarray, sr: int, path: str) -> None:
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "MP3 output requires the 'mp3' extra: pip install 'jam2song[mp3]'"
        )
    # pydub expects int16 interleaved bytes
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    if pcm.ndim == 2:
        # (channels, samples) → (samples, channels) → flatten to interleaved
        n_channels = pcm.shape[0]
        pcm = pcm.T.flatten()
    else:
        n_channels = 1
    seg = AudioSegment(
        pcm.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=n_channels,
    )
    seg.export(path, format="mp3")
