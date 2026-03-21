import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .analyzer import analyze_cached
from .arranger import arrange
from .classifier import classify
from .models import RenderParams
from .renderer import render
from .segmenter import segment
from .structures import list_structures, load_structure


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jam2song",
        description="Convert a jam session recording into a structured song.",
    )
    parser.add_argument("input", nargs="?", help="Path to input audio file")
    parser.add_argument(
        "-o", "--output",
        help="Output file path (.flac, .wav, .mp3). Cannot be used with multiple --structure flags.",
    )
    parser.add_argument(
        "--target-duration", type=float, default=210.0, metavar="SECS",
        help="Target song duration in seconds (default: 210, range: 60-600)",
    )
    parser.add_argument(
        "--structure", action="append", dest="structures", metavar="NAME_OR_PATH",
        help="Structure preset name or path to custom JSON (repeatable, default: loop_build_drop)",
    )
    parser.add_argument(
        "--list-structures", action="store_true",
        help="List available structure presets and exit",
    )
    parser.add_argument(
        "--crossfade", type=float, default=0.1, metavar="SECS",
        help="Crossfade duration between sections in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--fade-in", type=float, default=3.0, metavar="SECS",
        help="Fade-in duration at start in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--fade-out", type=float, default=5.0, metavar="SECS",
        help="Fade-out duration at end in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=0.5,
        help="Segmentation sensitivity 0.1-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--edl", metavar="EDL_PATH",
        help="Path to write the edit decision list JSON (single-structure only)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip reading and writing the analysis cache",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # --list-structures exits immediately
    if args.list_structures:
        names = list_structures()
        print("Available structure presets:")
        for name in names:
            print(f"  {name}")
        sys.exit(0)

    if not args.input:
        parser.error("input file is required")

    # Validate ranges
    if not (60 <= args.target_duration <= 600):
        parser.error("--target-duration must be between 60 and 600 seconds")
    if not (0.1 <= args.sensitivity <= 1.0):
        parser.error("--sensitivity must be between 0.1 and 1.0")

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {args.input}")

    # Apply default structure
    if not args.structures:
        args.structures = ["loop_build_drop"]

    # -o is ambiguous with multiple structures
    if args.output and len(args.structures) > 1:
        parser.error(
            "-o/--output cannot be used with multiple --structure flags; "
            "omit -o to use automatic output naming"
        )

    # --edl is only meaningful for a single structure
    if args.edl and len(args.structures) > 1:
        parser.error("--edl cannot be used with multiple --structure flags")

    # Load all structures up front so bad names fail before the slow analysis
    structures = []
    for name in args.structures:
        try:
            structures.append(load_structure(name))
        except FileNotFoundError:
            parser.error(f"Structure not found: {name}")

    # --- Analysis (runs once for all structures) ---
    print(f"Loading: {input_path.name}")

    analysis = analyze_cached(
        str(input_path), verbose=args.verbose, use_cache=not args.no_cache
    )
    info = analysis.audio_info
    chan_str = "stereo" if info.channels == 2 else (
        "mono" if info.channels == 1 else f"{info.channels}ch"
    )
    print(f"  ({_format_duration(info.duration)}, {info.sample_rate} Hz, {chan_str})")
    print(f"Detected tempo: {info.tempo_bpm:.1f} BPM")

    segments = segment(analysis, sensitivity=args.sensitivity)
    print(f"Segmentation: found {len(segments)} sections (sensitivity: {args.sensitivity})")

    segments, dist_matrix = classify(segments)

    if args.verbose:
        from collections import Counter
        import numpy as _np
        tiers = Counter(s.energy_tier for s in segments)
        trends = Counter(s.trend for s in segments)
        slopes = [s.energy_slope for s in segments]
        print(f"  Energy tiers: {dict(tiers)}")
        print(f"  Trends: {dict(trends)}")
        print(f"  Slope range: {min(slopes):.4f} to {max(slopes):.4f}, "
              f"mean={_np.mean(slopes):.4f}, std={_np.std(slopes):.4f}")

    # --- Per-structure arrange + render loop ---
    for structure in structures:
        output_path, edl_path = _resolve_output_paths(
            input_path, structure, args.output, args.target_duration, args.edl
        )

        render_params = RenderParams(
            crossfade=args.crossfade,
            fade_in=args.fade_in,
            fade_out=args.fade_out,
            output_path=output_path,
        )

        print(f"\nStructure: {structure.name} (target: {_format_duration(args.target_duration)})")

        plan = arrange(
            segments, dist_matrix, structure, info, args.target_duration, render_params,
            beat_times=analysis.beat_times,
        )

        role_similar = {s.role: s.similar_to for s in structure.sections}
        for arr in plan.arranged_sections:
            seg = arr.segment
            s_m, s_s = divmod(seg.source_start, 60)
            e_m, e_s = divmod(seg.source_end, 60)
            sim_note = f"  (similar to {role_similar[arr.role]})" if role_similar[arr.role] else ""
            print(
                f"  {arr.role:<12} ->  {int(s_m)}:{s_s:05.2f}-{int(e_m)}:{e_s:05.2f}"
                f"  ({arr.actual_duration:.1f}s)"
                f"  {seg.energy_tier}/{seg.trend:<8}"
                f"  score: {arr.score:.2f}"
                f"{sim_note}"
            )

        print(f"\nRendering with {args.crossfade}s crossfades...")
        output_duration = render(plan, analysis, output_path)

        print(f"Output: {output_path} ({_format_duration(output_duration)})")
        _write_edl(plan, output_path, output_duration, edl_path)
        print(f"EDL:    {edl_path}")


def _resolve_output_paths(
    input_path: Path,
    structure,
    explicit_output: "str | None",
    target_duration: float,
    explicit_edl: "str | None" = None,
) -> tuple[str, str]:
    """Return (wav_path, edl_path). Auto-names with versioning when no explicit output given."""
    if explicit_output:
        wav = Path(explicit_output)
        edl = Path(explicit_edl) if explicit_edl else wav.with_suffix("").with_suffix(".edl.json")
        return str(wav), str(edl)

    slug = structure.name.lower().replace(" ", "_")
    total_secs = int(round(target_duration))
    mins, secs = divmod(total_secs, 60)
    dur_str = f"{mins}m{secs:02d}s"
    base_dir = input_path.parent
    stem = input_path.stem

    version = 1
    while True:
        wav = base_dir / f"{stem}_{slug}_{dur_str}_v{version:02d}.flac"
        edl = wav.with_suffix("").with_suffix(".edl.json")
        if not wav.exists() and not edl.exists():
            break
        version += 1

    return str(wav), str(edl)


def _format_duration(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{int(s):02d}"


def _write_edl(plan, output_path: str, output_duration: float, edl_path: str) -> None:
    sections_data = []
    for arr in plan.arranged_sections:
        seg = arr.segment
        sections_data.append({
            "role": arr.role,
            "source_start": round(seg.source_start, 3),
            "source_end": round(seg.source_end, 3),
            "duration": round(arr.actual_duration, 3),
            "energy": round(seg.mean_energy, 4),
            "brightness": round(seg.mean_brightness, 4),
            "score": round(arr.score, 4),
            "score_breakdown": {
                "energy_fit": round(arr.score_breakdown.energy_fit, 4),
                "variety": round(arr.score_breakdown.variety, 4),
                "duration_fit": round(arr.score_breakdown.duration_fit, 4),
            },
        })

    edl = {
        "version": __version__,
        "input": {
            "path": plan.audio_info.path,
            "duration": round(plan.audio_info.duration, 3),
            "sample_rate": plan.audio_info.sample_rate,
            "channels": plan.audio_info.channels,
            "detected_tempo_bpm": round(plan.audio_info.tempo_bpm, 2),
        },
        "structure": {
            "name": plan.structure.name,
            "target_duration": plan.target_duration,
        },
        "sections": sections_data,
        "render": {
            "crossfade": plan.render_params.crossfade,
            "fade_in": plan.render_params.fade_in,
            "fade_out": plan.render_params.fade_out,
            "output_duration": round(output_duration, 3),
            "output_path": output_path,
        },
    }

    with open(edl_path, "w", encoding="utf-8") as f:
        json.dump(edl, f, indent=2)


if __name__ == "__main__":
    main()
