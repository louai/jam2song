# jam2song

Convert a long jam session recording into a structured 3–5 minute song automatically.

Give it a continuous 15–60 minute recording and it analyses the audio, finds the best segments, and assembles them into a song following a chosen structure template (verse/chorus, build/drop, etc.) with phrase-aligned cuts and smooth crossfades.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd jam2song
uv sync
```

No other system dependencies — ffmpeg is bundled automatically via `imageio-ffmpeg`.

## Usage

```bash
uv run jam2song my_jam.wav
```

This produces auto-named Opus files next to the source (e.g. `my_jam_loop_build_drop_3m30s_v01.opus`) plus matching `.edl.json` files.

### Options

```
uv run jam2song [input] [options]

positional:
  input                     Path to input audio (WAV, MP3, FLAC, OGG, AIFF, …)

options:
  -o, --output OUTPUT       Output path (.opus, .flac, .wav, .mp3; default: auto-named .opus next to source)
  --target-duration SECS    Target length in seconds (default: 210, range: 60–600)
  --structure NAME_OR_PATH  Built-in preset name or path to custom JSON (repeatable; default: loop_build_drop)
  --list-structures         Print available presets and exit
  --no-cache                Skip reading and writing the analysis cache
  --crossfade SECS          Crossfade between sections (default: 0.1)
  --fade-in SECS            Fade-in at start (default: 3.0)
  --fade-out SECS           Fade-out at end (default: 5.0)
  --sensitivity FLOAT       Segmentation sensitivity 0.1–1.0 (default: 0.5)
                            Lower = fewer, longer sections; higher = more, shorter sections
  --edl EDL_PATH            Path for the edit decision list JSON
  --verbose                 Show candidate scores and feature details
```

### Examples

```bash
# Render all three built-in structures in one pass (analysis runs once)
uv run jam2song session.wav --structure loop_build_drop --structure verse_chorus --structure highlight_reel

# 4-minute verse/chorus with explicit output path
uv run jam2song session.wav --target-duration 240 --structure verse_chorus -o my_song.wav

# Use a custom structure template
uv run jam2song session.wav --structure my_template.json

# Force fresh analysis, skip cache
uv run jam2song session.wav --no-cache

# List available presets
uv run jam2song --list-structures
```

## Built-in Structures

### `loop_build_drop` (default)

Classic electronic/loop music arc.

| Role | Energy | Notes |
|------|--------|-------|
| intro | low | Anchored to earliest source material |
| intro_fill | rising | Second intro segment — captures the drum fill leading into the first drop |
| build_1 | rising | |
| drop_1 | high | |
| breakdown | low | |
| build_2 | rising | |
| drop_2 | high | Similar to drop_1 — picks a spectrally close segment |
| outro | falling | Anchored to latest source material |

### `verse_chorus`

Traditional song form with repeated choruses.

| Role | Energy | Notes |
|------|--------|-------|
| intro | low | Anchored to source start |
| intro_fill | rising | |
| verse_1 | mid | |
| chorus_1 | high | |
| verse_2 | mid | |
| chorus_2 | high | Similar to chorus_1 |
| bridge | mid | |
| chorus_3 | high | Similar to chorus_1 |
| outro | falling | Anchored to source end |

### `highlight_reel`

Alternating peaks and valleys — a greatest-hits cut.

| Role | Energy |
|------|--------|
| moment_1 | high |
| breath_1 | low |
| moment_2 | high |
| breath_2 | mid |
| moment_3 | high |
| cooldown | falling |

## Custom Structure Templates

Create a JSON file with this schema:

```json
{
  "name": "my_structure",
  "description": "Optional description",
  "sections": [
    {"role": "intro",   "energy": "low",    "relative_duration": 1},
    {"role": "verse_1", "energy": "mid",    "relative_duration": 2},
    {"role": "chorus",  "energy": "high",   "relative_duration": 1.5},
    {"role": "outro",   "energy": "falling","relative_duration": 1}
  ]
}
```

**Fields:**

- `role` — unique label; roles starting with `intro` are anchored to the earliest source segments, roles starting with `outro` to the latest
- `energy` — one of: `low`, `mid`, `high`, `rising`, `falling`
- `relative_duration` — proportional share of total target duration (positive number)
- `similar_to` — (optional) role name of an earlier section; picks a spectrally similar segment to create repetition

## Output

### Audio

Opus (192 kbps lossy) by default — high quality, compact, widely supported. Use `-o output.flac` for lossless, `-o output.wav` for uncompressed, or `-o output.mp3` for MP3.

### Edit Decision List (EDL)

JSON file documenting every decision:

```json
{
  "version": "0.1.0",
  "input": {"path": "...", "duration": 1847.3, "sample_rate": 44100, "channels": 2, "detected_tempo_bpm": 122.4},
  "structure": {"name": "loop_build_drop", "target_duration": 210},
  "sections": [
    {
      "role": "intro",
      "source_start": 0.0,
      "source_end": 17.76,
      "duration": 17.76,
      "energy": 0.18,
      "brightness": 0.29,
      "score": 0.91,
      "score_breakdown": {"energy_fit": 1.1, "variety": 1.0, "duration_fit": 0.84}
    }
  ],
  "render": {"crossfade": 0.1, "fade_in": 3.0, "fade_out": 5.0, "output_duration": 213.1, "output_path": "..."}
}
```

## How It Works

1. **Analyze** — loads audio twice: a mono 22 kHz copy for analysis, the original for rendering. Extracts RMS energy, spectral centroid, onset strength, a novelty curve, and a beat grid.

2. **Segment** — peak-picks the novelty curve to find moments of textural change. Boundaries snap to a **2-bar (8-beat) phrase grid** for loop-coherent cuts. A fill bonus weights boundaries that follow a burst of onset activity (drum fills), so cuts tend to land after fills rather than mid-phrase.

3. **Classify** — ranks segments by mean energy into tiers (low/mid/high) and assigns a trend (rising/falling/steady) from the RMS slope within each segment.

4. **Arrange** — for each role in the structure template, scores candidate segments on:
   - **Energy fit** — how well tier and trend match the role's energy spec
   - **Variety** — spectral distance from already-placed sections (prevents repetition); inverted to similarity when `similar_to` is set
   - **Duration fit** — how close natural length is to the slot's target duration
   - **Temporal fit** — whether the segment's position in the source matches the role's expected position in the song; weighted more heavily for shorter target durations
   - **Transition score** — rewards energy moving in the direction the template intends between adjacent roles (e.g. the breakdown after a drop should be genuinely lower energy than the drop)

   Roles starting with `intro` are hard-anchored to the earliest unused segments; `outro` roles to the latest, preserving the natural opening and closing of the recording.

5. **Render** — extracts each chosen segment from the original-quality audio, joins them with equal-power crossfades, applies fade-in/out, and writes the output WAV.

## Running Tests

```bash
uv run pytest tests/ -q
```
