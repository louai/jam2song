from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

EnergyTier = Literal["low", "mid", "high"]
Trend = Literal["rising", "falling", "steady"]
EnergySpec = Literal["low", "mid", "high", "rising", "falling"]


@dataclass
class AudioInfo:
    path: str
    duration: float       # seconds
    sample_rate: int
    channels: int
    tempo_bpm: float


@dataclass
class Segment:
    index: int
    source_start: float   # seconds in original audio
    source_end: float
    duration: float
    mean_energy: float    # normalized 0-1
    mean_brightness: float
    onset_density: float  # onset mass per second (raw; normalized in classifier)
    energy_slope: float   # linear regression slope of RMS over normalized time
    internal_variance: float
    energy_tier: EnergyTier = "mid"
    trend: Trend = "steady"


@dataclass
class SectionSpec:
    role: str
    energy: EnergySpec
    relative_duration: float
    similar_to: str | None = None


@dataclass
class Structure:
    name: str
    description: str
    sections: list[SectionSpec]


@dataclass
class ScoreBreakdown:
    energy_fit: float
    variety: float        # or similarity when similar_to is set
    duration_fit: float
    temporal_fit: float = 1.0     # how close to expected temporal position (1.0 = perfect)
    transition_score: float = 1.0  # energy moves in the right direction from previous role


@dataclass
class ArrangedSection:
    role: str
    segment: Segment
    target_duration: float   # seconds this slot should occupy
    actual_duration: float   # may be less if segment was too short
    score: float
    score_breakdown: ScoreBreakdown


@dataclass
class RenderParams:
    crossfade: float = 0.1
    fade_in: float = 3.0
    fade_out: float = 5.0
    output_path: str = ""


@dataclass
class SongPlan:
    audio_info: AudioInfo
    structure: Structure
    target_duration: float
    arranged_sections: list[ArrangedSection]
    render_params: RenderParams


@dataclass
class AnalysisResult:
    audio_info: AudioInfo
    # Analysis copy (22050 Hz mono) — used for feature extraction
    y_analysis: object   # np.ndarray
    sr_analysis: int
    # Render copy (original quality) — used for final output
    y_render: object     # np.ndarray
    sr_render: int
    # Feature arrays (all normalized 0-1, frame-aligned at hop_length)
    hop_length: int
    beat_frames: object  # np.ndarray of frame indices
    beat_times: object   # np.ndarray of seconds
    tempo_bpm: float
    rms: object          # np.ndarray shape (n_frames,)
    spectral_centroid: object
    onset_strength: object
    novelty_curve: object
