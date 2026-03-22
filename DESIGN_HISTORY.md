# jam2song — Design History

A record of the key prompts and decisions that shaped this application. Smaller implementation fixes are omitted; this focuses on moments where the design meaningfully changed.

---

## 1. Initial Concept

> *"A Python CLI tool that takes a long, continuous jam session recording (15–60 minutes) and automatically edits it into a structured 3–5 minute song."*

The core premise: a jam session has no silence, no structure, just continuous music. The tool's job is to find the hidden structure within it. This ruled out silence-based segmentation from the start and established novelty detection (spectral flux) as the segmentation primitive.

The pipeline architecture was defined upfront: Analyze → Segment → Classify → Arrange → Render. Each phase has a single responsibility, making the system auditable — the EDL (Edit Decision List) JSON documents every decision so you can see exactly what was picked and why.

**Key decisions:**
- Novelty-curve segmentation, not silence detection
- Dual audio load: 22 kHz mono analysis copy + original quality render copy
- Structure templates as external JSON, not hardcoded logic
- EDL output for full transparency

---

## 2. No System Dependencies

> *"Get a portable ffmpeg, there should be no system dependencies like that."*

The original spec required ffmpeg to be installed on the system. This was replaced with `imageio-ffmpeg`, which bundles a static ffmpeg binary. At startup, `analyzer.py` calls `imageio_ffmpeg.get_ffmpeg_exe()` and prepends its directory to `PATH` so librosa picks it up transparently.

**Result:** `uv sync` is the entire install process. No system packages, no PATH configuration, works on any machine.

---

## 3. Output Sounded Terrible — Phrase-Aligned Cuts

> *"It sounds pretty terrible. There's lots of silences."*
>
> *"The cuts between the segments I'd like to sound more natural. It should be possible since the entire jam is on beat with repeated samples — cutting at consistent measures should negate the need for a crossfade, or at least a much shorter one. It would be extremely cool to cut segments after drum fills."*

The first working version was cutting at individual beat boundaries detected by peak-picking. This produced short, musically incoherent segments (2–3 seconds) with awkward transitions.

Two changes fixed this:

**Phrase-aligned cuts:** Boundaries now snap to a **2-bar (8-beat) phrase grid** instead of individual beats. In loop-based music every phrase is a complete musical thought; cutting at phrase boundaries means segments always start and end at a natural loop point, eliminating the need for long crossfades.

**Fill bonus:** A causal moving-average of onset strength over the preceding 2 beats is added to the novelty curve (weight 0.3). This biases segmentation boundaries to land *after* drum fills rather than mid-phrase — cuts feel like intentional transitions rather than arbitrary splices.

**Default crossfade reduced from 2.0s → 0.1s:** With phrase-aligned cuts, a long crossfade would blur the natural loop points. 0.1s is enough to prevent clicks at a hard splice while preserving the attack of the incoming phrase.

---

## 4. Natural Intro and Outro — The intro*/outro* Convention

> *"I want the generated song to have an intro and outro that more closely resembles the source. The source has natural intros and outros — the middle part of the source is what needs the most structural changes."*
>
> *"The drum intro fill is still missing from the song intro. There could be two intro segments — one for the beginning and another that starts at the drum intro fill."*

The original arranger treated `intro` and `outro` like any other role, selecting by energy score. This meant the actual start and end of the recording were often ignored in favour of higher-scoring mid-jam segments.

**Anchoring by role prefix:** Any role whose name starts with `"intro"` is hard-anchored to the earliest unused segments (sequential from source start). Any role starting with `"outro"` uses the latest segments working backwards. This is matched by `startswith()` so `intro_fill`, `intro_2`, etc. all follow the same rule.

**Two-part intro pattern:** Adding `intro_fill` as a second intro role — with `"energy": "rising"` — lets the structure capture both the quiet opening moments and the energetic fill that leads into the first drop. Both use their natural duration rather than being trimmed to a slot target.

**Fade-in suppression:** When the first section starts at source position 0, the fade-in is capped at 50 ms. A 3-second fade-in was silencing the drum fill that opens the recording.

**Result:** The rendered song now opens with the actual beginning of the jam and closes with its actual ending, with the fill intact.

---

## 5. Temporal Ordering

> *"I'd generally favour sequencing the structure in temporal order when it's possible. Shorter song durations should probably be more temporally accurate to the source, longer durations could have more variation — but temporal order should be weighted heavier. How is sequencing currently determined?"*

Before this change, sections were scored purely on energy fit, variety, and duration — with no awareness of where in the source recording a segment came from. A `bridge` role could pull a segment from minute 7 even when it was placed after a `verse_2` segment from minute 24.

**Expected fraction scoring:** Each non-anchor role is assigned an expected position `(k+1)/(M+1)` where `k` is its index among non-anchor roles and `M` is the total count. A `temporal_fit` score rewards segments whose source position matches this expected fraction.

**Duration-scaled weight:** The temporal weight is `0.5 * max(0, 1 - (target - 60) / 540)`. At 60s the temporal constraint is at full strength (0.5); at 210s it's ~0.36; at 600s it relaxes to zero. Short songs need to feel like a tight edit of the original; longer arrangements have room to jump around.

---

## 6. Energy Flow — Transition Scoring and Stronger Rising Matching

> *"Does the energy flow from low to high, and go from rising to falling over the course of the song? How is energy prioritised in the flow of the structure?"*

Two gaps in the energy scoring were identified:

**Weak rising matching:** A quiet segment with a rising RMS slope scored 0.75 for a `rising` spec — nearly as well as a mid-energy rising segment (0.90). Build sections were ending up with barely-audible material that didn't feel like a build. The low-tier score for `rising` was reduced from 0.5 → 0.2, making mid/rising the clear preference.

**No transition awareness:** Each role was scored in isolation. There was no reward for a candidate actually moving energy in the direction the template intended. A `breakdown` after a `drop` should prefer a segment that is genuinely lower-energy than the drop, not just a segment with a low `energy_tier` label.

**Transition scoring:** An energy level map `{low: 0.0, mid: 1.0, high: 2.0, rising: 1.5, falling: 0.5}` computes the intended direction of change between adjacent roles. The transition score uses `tanh(Δenergy × 10)` — smooth, bounded ±1, meaningful at small deltas. Weight: 0.15 (significant but not dominating).

**Result:** The energy arc now follows the template's intent not just in label-matching but in the actual amplitude relationships between adjacent sections.

---

## 7. Acceptance Test Structure

> *"Output a file per template and name it something short and friendly and versioned — do this for every non-trivial code change as an acceptance test."*
>
> *"Use the long template name and include the duration in minutes in the filenames too, put the version number at the end."*
>
> *"Create a dir structure for the acceptance tests similar to: `sample_input / song1 / source.wav / output / sourceName_templateName_duration_versionNumber`"*
>
> *"The input wav file should always retain its original name."*

Acceptance testing evolved from flat files in the project root to a structured layout under `sample_input/`. The key constraint: input files always keep their original filename (the date/ID is meaningful), outputs follow a consistent `sourceName_templateName_duration_vN` convention, and old versions are never deleted.

```
sample_input/
└── 01-260320_2143/
    ├── 01-260320_2143.wav          ← original name preserved
    └── output/
        ├── 01-260320_2143_loop_build_drop_3m46s_v6.wav
        ├── 01-260320_2143_verse_chorus_3m42s_v6.wav
        └── 01-260320_2143_highlight_reel_2m57s_v6.wav
```

---

## 8. Analysis Cache, Multi-Structure Rendering, and Smart Output Naming

> *"Create a cache file that contains all the analysis needed to generate new outputs so that subsequent generations are faster. The CLI should accept a list of templates to output to. Also do the todo about filenames when an explicit output path isn't given."*

Three quality-of-life features motivated by the iterative acceptance test workflow — running all three templates after every change was slow because analysis (librosa beat tracking, RMS, spectral centroid, novelty curve) runs on the full recording every time.

**Analysis cache** (`cache.py`): after the first run, analysis results (both audio copies + all feature arrays) are saved to `<source_stem>.cache.npz` and `<source_stem>.cache.json` next to the source file. On subsequent runs the cache is loaded instead of re-analysing, reducing a multi-second decode + feature extraction to a fast numpy load. The cache is invalidated by source file `mtime_ns` or `size` change. Writes are atomic (temp file + `os.replace()`). A `--no-cache` flag bypasses it entirely.

**Multi-structure rendering**: `--structure` now uses `action="append"` so it can be repeated. Analysis and segmentation run once; the arrange + render loop runs once per structure. All three templates in a single invocation:

```bash
uv run jam2song session.wav --structure loop_build_drop --structure verse_chorus --structure highlight_reel
```

`-o` and `--edl` are blocked when multiple `--structure` flags are given — they're ambiguous with multiple outputs.

**Smart default output naming**: when `-o` is omitted, outputs are written next to the source file as `<source_stem>_<template>_<duration>_v<NN>.opus`, auto-incrementing the version number to avoid overwriting previous renders. This makes the acceptance test workflow require no explicit path management — just run the tool and new versioned files appear alongside the source.

---

## 9. `condensed_jam` Structure — Long-form Granular Energy Arc

> *"I want a structure designed for a longer duration, around 10 minutes, that allows for a more granular progression of energy, similar to a condensed jam session."*

The existing templates (6–9 sections) were designed for 3–5 minute songs. At a 10-minute target, each section would be 45–100 seconds — too coarse for a gradual energy arc. A jam session has a natural shape: tentative opening, finding a groove, building to a peak, pulling back to explore, building again bigger, hitting a climax, then winding down.

**`condensed_jam` (18 sections)** mirrors this arc with fine-grained energy control:

- **3 peaks** (peak_1, peak_2, climax) with increasing intensity — the climax gets the largest `relative_duration` (2.0) so it dominates the arrangement
- **Exploratory middle** (explore_1 + explore_2) — a quiet, varied section between the two build/peak cycles that prevents the structure from feeling mechanical
- **`similar_to` callbacks** — groove_2 references groove_1 and peak_2 references peak_1, creating thematic repetition across the two halves
- **Gradual comedowns** via `"falling"` energy spec after each peak — the transition scoring ensures candidates actually drop in energy from the preceding peak
- **Short intro/outro weights** (0.8) — at 10 minutes the body matters more than bookends

At 600s target with 18 sections, each slot averages ~35 seconds — close to the natural segment lengths from phrase-aligned segmentation, giving good duration fit scores.
