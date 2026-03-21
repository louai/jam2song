# jam2song — Technical Specification

## Overview

A Python CLI tool that takes a long, continuous jam session recording (15–60 minutes) and automatically edits it into a structured 3–5 minute song. The input audio has **no silence or breaks** — it is continuous musical material throughout. The tool must analyze the audio to understand its musical content, identify the best segments, and arrange them into a configurable song structure with smooth transitions.

## Project Setup

### Python Environment

Use **modern Python tooling**:

- **Python 3.12+**
- **uv** for dependency management and virtual environments (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `pyproject.toml` as the single source of project metadata (no `setup.py`, no `requirements.txt`)
- A `src/jam2song/` layout (src-based project structure)
- Entry point defined in `pyproject.toml` so the tool installs as `jam2song` CLI command

### Project Structure

```
jam2song/
├── pyproject.toml
├── README.md
├── jam2song-spec.md
├── src/
│   └── jam2song/
│       ├── __init__.py
│       ├── __main__.py        # CLI entry point (argparse)
│       ├── analyzer.py        # Phase 1–2: audio loading + feature extraction
│       ├── segmenter.py       # Phase 3: structural segmentation at novelty boundaries
│       ├── classifier.py      # Phase 4: section classification by energy/character
│       ├── arranger.py        # Phase 5: selection + arrangement against structure template
│       ├── renderer.py        # Phase 6: assembly with crossfades + export
│       ├── structures.py      # Structure template loading, validation, scaling
│       ├── models.py          # Dataclasses: Segment, SongPlan, SectionSpec, etc.
│       └── structures/        # Built-in structure presets (JSON)
│           ├── loop_build_drop.json
│           ├── verse_chorus.json
│           └── highlight_reel.json
└── tests/
    ├── test_structures.py     # Template loading, validation, proportional scaling
    ├── test_segmenter.py
    └── test_arranger.py
```

### Dependencies

```toml
[project]
name = "jam2song"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "librosa>=0.10",
    "soundfile>=0.12",
    "numpy>=1.26",
    "scipy>=1.12",
    "imageio-ffmpeg>=0.6.0",
]

[project.scripts]
jam2song = "jam2song.__main__:main"
```

**No system ffmpeg required.** `imageio-ffmpeg` bundles a static ffmpeg binary. At startup `analyzer.py` calls `imageio_ffmpeg.get_ffmpeg_exe()` and prepends its directory to `os.environ["PATH"]` so librosa picks it up automatically.

---

## Pipeline

### Phase 1 — Load & Prep (`analyzer.py`)

- Bootstrap ffmpeg from the `imageio-ffmpeg` bundle before any audio loading
- Accept any audio format ffmpeg can decode (WAV, MP3, FLAC, OGG, AIFF, etc.), mono or stereo, any sample rate
- Load the audio twice:
  - **Analysis copy**: mono, resampled to 22050 Hz (for librosa feature extraction)
  - **Render copy**: original sample rate, original channels (for final output quality)
- Log: filename, detected duration, sample rate, channels

### Phase 2 — Feature Extraction (`analyzer.py`)

Compute the following time-series features across the full recording using a consistent hop length (512 samples):

| Feature | Library Call | Purpose |
|---------|-------------|---------|
| **Beat positions** | `librosa.beat.beat_track()` | Safe cut grid — all cuts snap to nearest beat |
| **Tempo (BPM)** | `librosa.beat.beat_track()` | Informational, logged to console and EDL |
| **RMS energy** | `librosa.feature.rms()` | Loudness envelope |
| **Spectral centroid** | `librosa.feature.spectral_centroid()` | Brightness / timbral character |
| **Onset strength** | `librosa.onset.onset_strength()` | Activity / density |
| **Novelty curve** | Derived from spectral flux of `librosa.stft()` | Detects moments of change — used for segmentation |

All features normalized to 0–1 range for comparable scoring.

### Phase 3 — Structural Segmentation (`segmenter.py`)

Since there are no silences or breaks, segmentation is driven by the **novelty curve**:

1. Compute novelty curve (spectral flux) across the full recording
2. Compute a **fill bonus** from the onset strength envelope: for each frame, `fill_bonus[i] = mean(onset_strength[i-lookback : i])` where `lookback` covers the preceding 2 beats. The combined signal is `novelty + 0.3 * fill_bonus`, normalized. This biases cut points to land **after drum fills** rather than mid-phrase.
3. **Peak-pick** the combined curve using `scipy.signal.find_peaks()` with configurable `prominence` and `distance` parameters (controlled by `--sensitivity`)
4. **Snap each boundary to the nearest 2-bar (8-beat) phrase boundary** from the beat grid (Phase 2). This guarantees loop-coherent cuts — segments always contain a whole number of phrases.
5. Hard minimum: no segment shorter than 5 seconds
6. This yields **variable-length sections** — each one represents a stretch of the jam with relatively consistent character (a groove, a breakdown, a build, etc.)
7. Each section gets a feature summary computed from the frames within it:
   - `mean_energy`: average RMS
   - `mean_brightness`: average spectral centroid
   - `onset_density`: onsets per second
   - `energy_slope`: linear regression slope of RMS over normalized time within the section (positive = building, negative = falling, near-zero = steady)
   - `internal_variance`: variance of RMS within the section (high = dynamic, low = consistent)

### Phase 4 — Section Classification (`classifier.py`)

Assign each detected section a character profile based on its features:

- **Energy tier**: Rank all sections by `mean_energy`, then divide:
  - Bottom 25% → `low`
  - Middle 50% → `mid`
  - Top 25% → `high`
- **Trend**: Based on `energy_slope`:
  - Significantly positive → `rising`
  - Significantly negative → `falling`
  - Otherwise → `steady`
- **Spectral distance matrix**: Compute pairwise distances between sections using a feature vector of `[mean_energy, mean_brightness, onset_density]`. Used by the arranger to maximize variety and find similar sections.

### Phase 5 — Selection & Arrangement (`arranger.py`)

This is the core creative logic. Given a **structure template** and a **target duration**, select the best section from the jam for each role in the template.

#### Structure Template Format

Structure templates are JSON files with this schema:

```json
{
  "name": "loop_build_drop",
  "description": "Classic electronic/loop music arc: build tension, release, repeat",
  "sections": [
    {"role": "intro",      "energy": "low",     "relative_duration": 1},
    {"role": "intro_fill", "energy": "rising",  "relative_duration": 1},
    {"role": "build_1",    "energy": "rising",  "relative_duration": 1.5},
    {"role": "drop_1",     "energy": "high",    "relative_duration": 2},
    {"role": "breakdown",  "energy": "low",     "relative_duration": 1},
    {"role": "build_2",    "energy": "rising",  "relative_duration": 1.5},
    {"role": "drop_2",     "energy": "high",    "relative_duration": 2, "similar_to": "drop_1"},
    {"role": "outro",      "energy": "falling", "relative_duration": 1}
  ]
}
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `role` | yes | Unique label for this section within the template |
| `energy` | yes | Energy profile to match: `low`, `mid`, `high`, `rising`, `falling` |
| `relative_duration` | yes | Proportional share of total duration (unitless ratio, must be > 0) |
| `similar_to` | no | Role name of a **previously defined** section — arranger picks a segment spectrally similar to that one. Creates repetition/hook effect. Cannot be a forward reference or self-reference. |

#### intro*/outro* Role Prefix Convention

Any role whose name starts with `"intro"` (`intro`, `intro_fill`, `intro_2`, …) is **anchored to the start of the recording**: the arranger picks sequentially from the earliest unused segments. This preserves the natural opening of the jam including drum fills and build-ups.

Any role starting with `"outro"` is anchored to the end: segments are picked working backwards from the latest source position.

Intro/outro sections use their **natural duration** (capped at 2× the computed slot target) rather than being trimmed to the slot target — this respects the actual shape of the recording's opening and closing.

**Fade-in suppression**: if the first section starts at source position 0, the fade-in is suppressed to 50 ms (rather than the full `--fade-in` duration) so that a drum fill or immediate musical content at the very start of the recording is not attenuated.

#### Duration Scaling

1. Sum all `relative_duration` values → `total_weight`
2. `effective_target = target_duration + crossfade * (num_sections - 1)` — crossfades overlap so we need slightly more source material than the final duration
3. For each section: `target_seconds = (relative_duration / total_weight) * effective_target`

#### Selection Algorithm

For each non-anchor section in the template, in order:

1. **Filter candidates** by energy match:
   - `low` → sections with `energy_tier == "low"`
   - `mid` → sections with `energy_tier == "mid"`
   - `high` → sections with `energy_tier == "high"`
   - `rising` → sections with `trend == "rising"`
   - `falling` → sections with `trend == "falling"`
   - Fallback: if no matching candidates, use any unused section

2. **Score each candidate** (multiple factors):

   | Factor | Weight | Description |
   |--------|--------|-------------|
   | Energy fit | 1× | How well tier/trend matches the role's energy spec (see below) |
   | Variety | 1× | Average spectral distance from already-placed sections. Inverted to similarity if `similar_to` is set. |
   | Duration fit | 1× | `min(seg.duration, target) / max(seg.duration, target)` |
   | Temporal fit | `temporal_weight` | How close the segment's source position is to the role's expected position. `expected_fraction = (k+1)/(M+1)` where k is the role's index among non-anchor roles and M is total non-anchor roles. |
   | Transition score | 0.15 | Rewards energy moving in the direction the template intends between adjacent roles (see below). |

   **Score formula:**
   ```
   score = (energy_fit + variety + duration_fit
            + temporal_weight * temporal_fit
            + 0.15 * transition_score
           ) / (3 + temporal_weight + 0.15)
           + positional_bonus
   ```

   **Temporal weight** scales inversely with target duration: tighter constraint for short songs, relaxed for long ones:
   ```
   temporal_weight = 0.5 * max(0.0, 1.0 - (target_duration - 60) / (600 - 60))
   ```
   At 60s → weight = 0.5; at 210s → ~0.36; at 600s → 0.0.

3. **Energy fit scoring** (`_energy_fit`):
   - `low` spec: low tier = 1.0, mid = 0.4, high = 0.0; ×1.1 bonus if `steady` trend
   - `mid` spec: low = 0.3, mid = 1.0, high = 0.3
   - `high` spec: low = 0.0, mid = 0.4, high = 1.0; ×1.1 bonus if `steady` trend
   - `rising` spec: `(trend_score + tier_score) / 2` where trend_score = {rising:1.0, steady:0.3, falling:0.0}, tier_score = {low:0.2, mid:0.8, high:1.0} — low-tier rising segments score only 0.60 to avoid using barely-audible material for build sections
   - `falling` spec: `(trend_score + tier_score) / 2` where trend_score = {falling:1.0, steady:0.3, rising:0.0}, tier_score = {high:0.5, mid:0.8, low:1.0}

4. **Transition scoring** between adjacent roles:
   - Map energy specs to levels: `{low: 0.0, mid: 1.0, high: 2.0, rising: 1.5, falling: 0.5}`
   - `direction = ENERGY_LEVEL[spec.energy] - ENERGY_LEVEL[prev_spec.energy]`
   - `diff = candidate.mean_energy - prev_segment.mean_energy`
   - If direction > 0 (template expects energy increase): `score = 0.5 + 0.5 * tanh(diff * 10)`
   - If direction < 0 (template expects energy decrease): `score = 0.5 - 0.5 * tanh(diff * 10)`
   - If direction == 0 (same level): `score = 1.0`
   - `tanh(diff * 10)` is smooth and bounded ±1; at Δenergy = ±0.1 the score is ~0.88/0.12

5. **Positional bonus** (soft tiebreaker, added after normalization):
   - `intro*` role and segment within first 25% of jam: +0.15
   - `outro*` role and segment within last 25% of jam: +0.20
   - Non-intro role and segment at source_start == 0: -0.15 (penalises using recording startup noise as a groove section)

6. **Select the highest-scoring candidate** that hasn't already been used

7. **Trim** the selected section to its target duration, snapping to the nearest beat that falls at or before the target end time (beat-snapped duration). If the section is shorter than the target, use its natural length.

8. **No segment reuse**: once a segment is placed, it is excluded from all subsequent roles.

### Phase 6 — Render (`renderer.py`)

1. For each section in the arrangement plan, extract audio from the **original quality** render copy (not the 22050 Hz analysis copy)
2. Trim each extraction to beat-aligned boundaries
3. Join adjacent sections with **equal-power crossfades**:
   - Default crossfade duration: **0.1 seconds** (configurable via `--crossfade`). The short default is intentional — phrase-aligned cuts land on the beat grid and don't need long crossfades; 0.1s is enough to prevent clicks at a hard splice.
   - Equal-power curve: `gain_out = cos(t * π/2)`, `gain_in = sin(t * π/2)` where t goes from 0 to 1 over the crossfade duration
   - This maintains constant perceived loudness through the transition (unlike linear crossfades which dip)
4. Apply a **fade-in** at the start (default 3 seconds, configurable via `--fade-in`). Suppressed to 50 ms if the first section begins at source position 0 (preserves drum fills at the top of the recording).
5. Apply a **fade-out** at the end (default 5 seconds, configurable via `--fade-out`)
6. Write output file at original input sample rate and channel count. Format is determined by file extension:
   - **`.opus`** (default) — Opus at 192 kbps via bundled ffmpeg. Perceptually transparent for music, ~1/10th the size of WAV.
   - **`.flac`** — 24-bit FLAC, lossless. Via soundfile/libsndfile.
   - **`.wav`** — 24-bit PCM, uncompressed. Via soundfile.
   - **`.ogg`** — Vorbis at quality 8 via ffmpeg.
   - **`.aac`** / **`.m4a`** — AAC at 256 kbps via ffmpeg.
   - **`.mp3`** — via pydub (requires optional `mp3` dependency).

---

## CLI Interface

```
usage: jam2song [-h] [-o OUTPUT] [--target-duration SECONDS]
                [--structure NAME_OR_PATH] [--list-structures]
                [--crossfade SECONDS] [--fade-in SECONDS] [--fade-out SECONDS]
                [--sensitivity FLOAT] [--edl EDL_PATH]
                [--verbose]
                input

Convert a jam session recording into a structured song.

positional arguments:
  input                     Path to input audio file (any format ffmpeg supports)

options:
  -o, --output OUTPUT       Output file path (.opus, .flac, .wav, .mp3; default: auto-named .opus next to source)
  --target-duration SECS    Target song duration in seconds (default: 210, range: 60–600)
  --structure NAME_OR_PATH  Structure preset name or path to custom JSON (repeatable; default: loop_build_drop)
  --list-structures         List available structure presets and exit
  --crossfade SECS          Crossfade duration between sections in seconds (default: 0.1)
  --fade-in SECS            Fade-in duration at start in seconds (default: 3.0)
  --fade-out SECS           Fade-out duration at end in seconds (default: 5.0)
  --sensitivity FLOAT       Segmentation sensitivity: lower = fewer, longer sections;
                            higher = more, shorter sections (default: 0.5, range: 0.1–1.0)
  --edl EDL_PATH            Path to write the edit decision list JSON (single-structure only)
  --no-cache                Skip reading and writing the analysis cache
  --verbose                 Verbose logging (show energy tier/trend distribution, slope range)
```

### Examples

```bash
# Basic usage — outputs auto-named next to source with version number
uv run jam2song my_jam.wav

# Render all three built-in structures in one pass (analysis runs once)
uv run jam2song my_jam.wav --structure loop_build_drop --structure verse_chorus --structure highlight_reel

# Explicit output path, 4-minute target
uv run jam2song my_jam.wav --target-duration 240 --structure verse_chorus -o my_song.wav

# Custom structure file
uv run jam2song session.flac --structure ~/my_template.json

# Re-render without using cached analysis
uv run jam2song my_jam.wav --no-cache

# Just list what presets are available
uv run jam2song --list-structures
```

---

## Output

### Audio File

The primary output. Opus (192 kbps) by default — high quality, compact, widely supported. FLAC, WAV, OGG, AAC, or MP3 when the output path uses those extensions.

### Edit Decision List (EDL)

A JSON file written alongside the output audio (or at `--edl` path). Contains everything needed to understand and reproduce the edit:

```json
{
  "version": "0.1.0",
  "input": {
    "path": "my_jam.wav",
    "duration": 1847.3,
    "sample_rate": 44100,
    "channels": 2,
    "detected_tempo_bpm": 122.4
  },
  "structure": {
    "name": "loop_build_drop",
    "target_duration": 210
  },
  "sections": [
    {
      "role": "intro",
      "source_start": 0.0,
      "source_end": 17.76,
      "duration": 17.76,
      "energy": 0.18,
      "brightness": 0.29,
      "score": 0.91,
      "score_breakdown": {
        "energy_fit": 1.1,
        "variety": 1.0,
        "duration_fit": 0.84
      }
    }
  ],
  "render": {
    "crossfade": 0.1,
    "fade_in": 3.0,
    "fade_out": 5.0,
    "output_duration": 213.1,
    "output_path": "my_jam_song.wav"
  }
}
```

### Console Output

```
Loading: my_jam.wav
  (30:47, 44100 Hz, stereo)
Analyzing audio features...
Detected tempo: 122.4 BPM
Segmentation: found 24 sections (sensitivity: 0.5)
Structure: loop_build_drop (target: 3:30)

  intro        ->  0:00.00-0:17.76  (17.8s)  low/steady    score: 0.91
  intro_fill   ->  0:17.76-0:35.52 (17.8s)  rising/rising  score: 0.88
  build_1      ->  8:12.00-8:47.00 (35.2s)  mid/rising    score: 0.91
  drop_1       -> 12:33.00-13:35.0 (62.1s)  high/steady   score: 0.94
  breakdown    -> 18:02.00-18:24.0 (22.3s)  low/steady    score: 0.83
  build_2      -> 22:15.00-22:48.0 (33.1s)  mid/rising    score: 0.88
  drop_2       -> 13:35.00-14:33.0 (58.4s)  high/steady   score: 0.90  (similar to drop_1)
  outro        -> 28:44.00-29:05.0 (21.2s)  low/falling   score: 0.79

Rendering with 0.1s crossfades...
Output: my_jam_song.wav (3:33)
EDL: my_jam_song.edl.json
```

---

## Built-in Structure Presets

### `loop_build_drop` (default)

Classic electronic/loop music structure. Build tension, release at the drop, break down, repeat.

```json
{
  "name": "loop_build_drop",
  "description": "Classic electronic/loop music arc: build tension, release, repeat",
  "sections": [
    {"role": "intro",      "energy": "low",     "relative_duration": 1},
    {"role": "intro_fill", "energy": "rising",  "relative_duration": 1},
    {"role": "build_1",    "energy": "rising",  "relative_duration": 1.5},
    {"role": "drop_1",     "energy": "high",    "relative_duration": 2},
    {"role": "breakdown",  "energy": "low",     "relative_duration": 1},
    {"role": "build_2",    "energy": "rising",  "relative_duration": 1.5},
    {"role": "drop_2",     "energy": "high",    "relative_duration": 2, "similar_to": "drop_1"},
    {"role": "outro",      "energy": "falling", "relative_duration": 1}
  ]
}
```

### `verse_chorus`

Traditional song form. Alternating verse/chorus with a bridge for contrast.

```json
{
  "name": "verse_chorus",
  "description": "Traditional song structure with verses, choruses, and a bridge",
  "sections": [
    {"role": "intro",      "energy": "low",     "relative_duration": 1},
    {"role": "intro_fill", "energy": "rising",  "relative_duration": 1},
    {"role": "verse_1",    "energy": "mid",     "relative_duration": 2},
    {"role": "chorus_1",   "energy": "high",    "relative_duration": 1.5},
    {"role": "verse_2",    "energy": "mid",     "relative_duration": 2},
    {"role": "chorus_2",   "energy": "high",    "relative_duration": 1.5, "similar_to": "chorus_1"},
    {"role": "bridge",     "energy": "mid",     "relative_duration": 1},
    {"role": "chorus_3",   "energy": "high",    "relative_duration": 1.5, "similar_to": "chorus_1"},
    {"role": "outro",      "energy": "falling", "relative_duration": 1}
  ]
}
```

### `highlight_reel`

Simple alternating high/low energy. Good for showcasing the best moments.

```json
{
  "name": "highlight_reel",
  "description": "Alternating peaks and valleys — a greatest hits from the jam",
  "sections": [
    {"role": "moment_1",  "energy": "high",    "relative_duration": 2},
    {"role": "breath_1",  "energy": "low",     "relative_duration": 1},
    {"role": "moment_2",  "energy": "high",    "relative_duration": 2},
    {"role": "breath_2",  "energy": "mid",     "relative_duration": 1},
    {"role": "moment_3",  "energy": "high",    "relative_duration": 2},
    {"role": "cooldown",  "energy": "falling", "relative_duration": 1}
  ]
}
```

---

## Structure Template Validation

When loading a structure template (built-in or custom), validate:

1. `name` is a non-empty string
2. `sections` is a non-empty array
3. Each section has `role` (string), `energy` (one of: `low`, `mid`, `high`, `rising`, `falling`), and `relative_duration` (positive number)
4. All `role` values are unique within the template
5. If `similar_to` is present, it references a `role` that appears **earlier** in the sections list (no forward references, no self-references)
6. At least one section exists

On validation failure, raise `ValueError` with a clear message identifying which field failed.

---

## Data Models (`models.py`)

```python
@dataclass
class Segment:
    index: int
    source_start: float       # seconds in original audio
    source_end: float
    duration: float
    mean_energy: float        # normalized 0–1
    mean_brightness: float    # normalized 0–1
    onset_density: float      # onset mass per second (raw; normalized in classifier)
    energy_slope: float       # linear regression slope of RMS over normalized time
    internal_variance: float
    energy_tier: EnergyTier = "mid"   # "low" | "mid" | "high"
    trend: Trend = "steady"           # "rising" | "falling" | "steady"

@dataclass
class SectionSpec:
    role: str
    energy: EnergySpec        # "low" | "mid" | "high" | "rising" | "falling"
    relative_duration: float
    similar_to: str | None = None

@dataclass
class ScoreBreakdown:
    energy_fit: float
    variety: float            # or similarity when similar_to is set
    duration_fit: float
    temporal_fit: float = 1.0        # 1.0 = perfect match to expected position
    transition_score: float = 1.0    # 1.0 = energy moving in ideal direction

@dataclass
class ArrangedSection:
    role: str
    segment: Segment
    target_duration: float
    actual_duration: float    # may be less if segment was shorter than target
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
```

---

## Design Principles

- **No silence in input assumed**: The entire pipeline is designed around continuous audio. Segmentation uses novelty detection, not silence detection.
- **Phrase-aligned cuts**: Every cut point snaps to a 2-bar (8-beat) phrase grid. This is critical for musical coherence in loop-based material.
- **Fill-aware boundaries**: The segmentation novelty curve is boosted by a causal onset density window, so boundaries naturally prefer to land after drum fills.
- **Equal-power crossfades**: Maintain constant loudness through transitions.
- **Original quality rendering**: Analysis happens on a downsampled copy; the final output uses the original audio at its native sample rate and channel count.
- **Temporal coherence**: The arrangement respects the natural arc of the jam — intros come from the start, outros from the end, and middle sections are distributed temporally across the recording.
- **Energy arc**: Transition scoring rewards arrangements where energy actually moves in the direction the structure template intends, not just energy tier matching.
- **Deterministic**: Same input + same parameters = same output. No randomness in the selection algorithm.
- **Transparent**: The EDL JSON makes every decision auditable. You can see exactly what was picked and why.
- **No system dependencies**: ffmpeg is bundled via `imageio-ffmpeg`. `uv sync` is the entire install process.

---

## Acceptance Testing

After every non-trivial code change, render one output per built-in structure template as a listening test. Outputs live under `sample_input/` alongside their source recording.

### Directory Structure

```
sample_input/
└── <source_stem>/
    ├── <source_stem>.wav                                  ← original input, name never changed
    └── output/
        ├── <source_stem>_<template>_<duration>_v<N>.opus
        └── <source_stem>_<template>_<duration>_v<N>.edl.json
```

Example for source file `01-260320_2143.wav`:

```
sample_input/
└── 01-260320_2143/
    ├── 01-260320_2143.wav
    └── output/
        ├── 01-260320_2143_loop_build_drop_3m46s_v7.wav
        ├── 01-260320_2143_loop_build_drop_3m46s_v7.edl.json
        ├── 01-260320_2143_verse_chorus_3m42s_v7.wav
        ├── 01-260320_2143_verse_chorus_3m42s_v7.edl.json
        ├── 01-260320_2143_highlight_reel_2m57s_v7.wav
        └── 01-260320_2143_highlight_reel_2m57s_v7.edl.json
```

### Rules

- **Never rename the input file** — the original filename encodes meaningful metadata (date, session ID, etc.)
- **Never delete old output versions** — keep all previous renders for comparison
- Duration in the filename comes from the actual rendered output duration (read from the EDL), not the `--target-duration` flag
- Increment the version number with each new render after a code change
- Render commands:

```bash
src="sample_input/01-260320_2143/01-260320_2143.wav"
out="sample_input/01-260320_2143/output"

uv run jam2song "$src" --structure loop_build_drop  -o "$out/01-260320_2143_loop_build_drop_3m30s_v7.opus"
uv run jam2song "$src" --structure verse_chorus     -o "$out/01-260320_2143_verse_chorus_3m30s_v7.opus"
uv run jam2song "$src" --structure highlight_reel   -o "$out/01-260320_2143_highlight_reel_3m30s_v7.opus"
```

---

## Future Considerations (Not in v0.1)

These are noted for future iterations but are not currently implemented:

- ~~**Smart default output path**~~ *(implemented)*
- ~~**Analysis cache**~~ *(implemented)*
- **Web UI template editor**: browser-based interface showing all candidate segments per role, letting you swap candidates and audition each segment in isolation before committing to a final render
- Interactive mode (preview candidates, manually pick sections)
- Demucs stem separation integration for smarter crossfades
- Key detection and harmonic compatibility scoring between sections
- BPM-matched time-stretching for sections at slightly different tempos
- MP3 output (was in original spec; removed pending pydub integration)
