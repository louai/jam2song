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
├── structures/                # Built-in structure presets (JSON)
│   ├── loop_build_drop.json
│   ├── verse_chorus.json
│   └── highlight_reel.json
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
│       └── models.py          # Dataclasses: Segment, SongPlan, SectionSpec, etc.
└── tests/
    ├── test_structures.py     # Template loading, validation, proportional scaling
    ├── test_segmenter.py
    ├── test_arranger.py
    └── test_renderer.py       # Crossfade math, fade curves
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
]

[project.optional-dependencies]
mp3 = ["pydub>=0.25"]

[project.scripts]
jam2song = "jam2song.__main__:main"
```

External requirement: **ffmpeg** must be installed on the system (used by librosa as audio decoding backend).

---

## Pipeline

### Phase 1 — Load & Prep (`analyzer.py`)

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
2. **Peak-pick** the novelty curve to find moments of significant textural change. Use `scipy.signal.find_peaks()` with configurable `prominence` and `distance` parameters to control sensitivity
3. **Snap each boundary to the nearest beat** from the beat grid (Phase 2) for clean cuts
4. This yields **variable-length sections** — each one represents a stretch of the jam with relatively consistent character (a groove, a breakdown, a build, etc.)
5. Each section gets a feature summary computed from the frames within it:
   - `mean_energy`: average RMS
   - `mean_brightness`: average spectral centroid
   - `onset_density`: onsets per second
   - `energy_slope`: linear regression slope of RMS over time within the section (positive = building, negative = falling, near-zero = steady)
   - `internal_variance`: variance of RMS within the section (high = dynamic, low = consistent)

### Phase 4 — Section Classification (`classifier.py`)

Assign each detected section a character profile based on its features:

- **Energy tier**: Rank all sections by `mean_energy`, then divide into quartiles:
  - Bottom 25% → `low`
  - Middle 50% → `mid`
  - Top 25% → `high`
- **Trend**: Based on `energy_slope`:
  - Significantly positive → `rising`
  - Significantly negative → `falling`
  - Otherwise → `steady`
- **Spectral distance matrix**: Compute pairwise distances between sections using a feature vector of `[mean_energy, mean_brightness, onset_density]`. This is used later by the arranger to maximize variety and find similar sections.

### Phase 5 — Selection & Arrangement (`arranger.py`)

This is the core creative logic. Given a **structure template** and a **target duration**, select the best section from the jam for each role in the template.

#### Structure Template Format

Structure templates are JSON files with this schema:

```json
{
  "name": "loop_build_drop",
  "description": "Classic electronic/loop music arc: build tension, release, repeat",
  "sections": [
    {
      "role": "intro",
      "energy": "low",
      "relative_duration": 1
    },
    {
      "role": "build_1",
      "energy": "rising",
      "relative_duration": 1.5
    },
    {
      "role": "drop_1",
      "energy": "high",
      "relative_duration": 2
    },
    {
      "role": "breakdown",
      "energy": "low",
      "relative_duration": 1
    },
    {
      "role": "build_2",
      "energy": "rising",
      "relative_duration": 1.5
    },
    {
      "role": "drop_2",
      "energy": "high",
      "relative_duration": 2,
      "similar_to": "drop_1"
    },
    {
      "role": "outro",
      "energy": "falling",
      "relative_duration": 1
    }
  ]
}
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `role` | yes | Unique label for this section within the template |
| `energy` | yes | Energy profile to match: `low`, `mid`, `high`, `rising`, `falling` |
| `relative_duration` | yes | Proportional share of total duration (unitless ratio) |
| `similar_to` | no | Role name of a previously defined section — arranger picks a segment spectrally similar to that one. Creates repetition/hook effect. |

#### Duration Scaling

1. Sum all `relative_duration` values across sections → `total_weight`
2. For each section: `target_seconds = (section.relative_duration / total_weight) * target_duration`
3. Subtract estimated crossfade time from total: `effective_target = target_duration + (crossfade_duration * (num_sections - 1))` — since crossfades overlap, we need slightly more source material than the final duration

#### Selection Algorithm

For each section in the template, in order:

1. **Filter candidates** by energy match:
   - `low` → sections in bottom 25% energy, prefer `steady` trend
   - `mid` → sections in middle 50% energy
   - `high` → sections in top 25% energy, prefer `steady` trend
   - `rising` → sections with positive `energy_slope` (any energy tier)
   - `falling` → sections with negative `energy_slope` (any energy tier)

2. **Score each candidate** (three factors, equally weighted):
   - **Energy fit** (0–1): How well the section's energy tier matches the requested energy profile
   - **Variety** (0–1): Average spectral distance from all previously selected sections. Higher = more distinct = better (prevents picking the same groove twice for different roles)
   - **Duration fit** (0–1): How close the section's natural duration is to the target duration for this role. Prefer sections that need minimal trimming

3. **If `similar_to` is set**: Replace the Variety score with a **Similarity score** — spectral closeness to the referenced section. This inverts the logic: closer = better.

4. **Select the highest-scoring candidate** that hasn't already been used

5. **Trim or extend** the selected section to fit its target duration:
   - Trimming: Cut from the end, snapping to the nearest beat boundary
   - If the section is too short: Allow it at natural length (don't pad with silence). Adjust other sections' durations slightly to compensate

6. **Positional preference** (tiebreaker): For `intro` roles, prefer sections from the first 25% of the jam. For `outro` roles, prefer sections from the last 25%. This respects the natural arc of a jam session.

### Phase 6 — Render (`renderer.py`)

1. For each section in the arrangement plan, extract audio from the **original quality** render copy (not the 22050 Hz analysis copy)
2. Trim each extraction to beat-aligned boundaries
3. Join adjacent sections with **equal-power crossfades**:
   - Default crossfade duration: 2 seconds (configurable via `--crossfade`)
   - Equal-power curve: `gain_out = cos(t * π/2)`, `gain_in = sin(t * π/2)` where t goes from 0 to 1 over the crossfade duration
   - This maintains constant perceived loudness through the transition (unlike linear crossfades which dip)
4. Apply a **fade-in** at the start (default 3 seconds, configurable via `--fade-in`)
5. Apply a **fade-out** at the end (default 5 seconds, configurable via `--fade-out`)
6. Write output file:
   - Default: WAV (same sample rate and channels as input)
   - If output filename ends in `.mp3`: use pydub for MP3 encoding (requires `mp3` optional dependency)

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
  -o, --output OUTPUT       Output file path (default: input_stem_song.wav)
  --target-duration SECS    Target song duration in seconds (default: 210, range: 60–600)
  --structure NAME_OR_PATH  Structure preset name or path to custom JSON file (default: loop_build_drop)
  --list-structures         List available structure presets and exit
  --crossfade SECS          Crossfade duration between sections in seconds (default: 2.0)
  --fade-in SECS            Fade-in duration at start in seconds (default: 3.0)
  --fade-out SECS           Fade-out duration at end in seconds (default: 5.0)
  --sensitivity FLOAT       Segmentation sensitivity: lower = fewer, longer sections;
                            higher = more, shorter sections (default: 0.5, range: 0.1–1.0)
  --edl EDL_PATH            Path to write the edit decision list JSON (default: alongside output file)
  --verbose                 Verbose logging (show all candidate scores, feature details)
```

### Examples

```bash
# Basic usage with defaults
jam2song my_jam.wav

# Target a 4-minute song with the verse/chorus structure
jam2song my_jam.wav --target-duration 240 --structure verse_chorus

# Custom structure file, MP3 output
jam2song session.flac -o finished_track.mp3 --structure ~/my_template.json

# Longer crossfades, fewer detected sections
jam2song long_session.wav --crossfade 4 --sensitivity 0.3

# Just list what presets are available
jam2song --list-structures
```

---

## Output

### Audio File

The primary output. WAV by default at original input quality. MP3 if the output path ends in `.mp3`.

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
      "source_start": 45.2,
      "source_end": 66.1,
      "duration": 20.9,
      "energy": 0.23,
      "brightness": 0.31,
      "score": 0.87,
      "score_breakdown": {
        "energy_fit": 0.92,
        "variety": 0.81,
        "duration_fit": 0.88
      }
    }
  ],
  "render": {
    "crossfade": 2.0,
    "fade_in": 3.0,
    "fade_out": 5.0,
    "output_duration": 212.4,
    "output_path": "my_jam_song.wav"
  }
}
```

### Console Output

A human-readable summary printed during processing:

```
Loading: my_jam.wav (30:47, 44100 Hz, stereo)
Analyzing audio features...
Detected tempo: 122.4 BPM
Segmentation: found 24 sections (sensitivity: 0.5)
Structure: loop_build_drop (target: 3:30)

  intro      →  0:45–1:06  (20.9s)  energy: low     score: 0.87
  build_1    →  8:12–8:47  (35.2s)  energy: rising  score: 0.91
  drop_1     → 12:33–13:35 (62.1s)  energy: high    score: 0.94
  breakdown  → 18:02–18:24 (22.3s)  energy: low     score: 0.83
  build_2    → 22:15–22:48 (33.1s)  energy: rising  score: 0.88
  drop_2     → 13:35–14:33 (58.4s)  energy: high    score: 0.90  (similar to drop_1)
  outro      → 28:44–29:05 (21.2s)  energy: falling score: 0.79

Rendering with 2.0s crossfades...
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
    {"role": "intro",     "energy": "low",     "relative_duration": 1},
    {"role": "build_1",   "energy": "rising",  "relative_duration": 1.5},
    {"role": "drop_1",    "energy": "high",    "relative_duration": 2},
    {"role": "breakdown", "energy": "low",     "relative_duration": 1},
    {"role": "build_2",   "energy": "rising",  "relative_duration": 1.5},
    {"role": "drop_2",    "energy": "high",    "relative_duration": 2, "similar_to": "drop_1"},
    {"role": "outro",     "energy": "falling", "relative_duration": 1}
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
    {"role": "intro",     "energy": "low",     "relative_duration": 1},
    {"role": "verse_1",   "energy": "mid",     "relative_duration": 2},
    {"role": "chorus_1",  "energy": "high",    "relative_duration": 1.5},
    {"role": "verse_2",   "energy": "mid",     "relative_duration": 2},
    {"role": "chorus_2",  "energy": "high",    "relative_duration": 1.5, "similar_to": "chorus_1"},
    {"role": "bridge",    "energy": "mid",     "relative_duration": 1},
    {"role": "chorus_3",  "energy": "high",    "relative_duration": 1.5, "similar_to": "chorus_1"},
    {"role": "outro",     "energy": "falling", "relative_duration": 1}
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
5. If `similar_to` is present, it references a `role` that appears **earlier** in the sections list
6. At least one section exists (degenerate but valid: single-section template just extracts the best segment)

On validation failure, print a clear error message identifying which field failed and exit.

---

## Design Principles

- **No silence in input assumed**: The entire pipeline is designed around continuous audio. Segmentation uses novelty detection, not silence detection.
- **Beat-aligned cuts**: Every cut point snaps to the nearest detected beat. This is critical for musical coherence.
- **Equal-power crossfades**: Maintain constant loudness through transitions.
- **Original quality rendering**: Analysis happens on a downsampled copy; the final output uses the original audio at its native sample rate and channel count.
- **Deterministic**: Same input + same parameters = same output. No randomness in the selection algorithm (scores are deterministic).
- **Transparent**: The EDL JSON makes every decision auditable. You can see exactly what was picked and why.

---

## Future Considerations (Not in v0.1)

These are noted for future iterations but should NOT be implemented in the initial build:

- Interactive mode (preview candidates, manually pick sections)
- Demucs stem separation integration for smarter crossfades
- Key detection and harmonic compatibility scoring between sections
- BPM-matched time-stretching for sections at slightly different tempos
- Web UI for visual waveform editing of the arrangement
