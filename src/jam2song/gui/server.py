"""
Local HTTP server for the jam2song arrangement editor GUI.

Serves the single-page HTML frontend and exposes API endpoints for
audio playback, waveform data, and arrangement editing/re-rendering.
"""

import io
import json
import os
import threading
import webbrowser
from dataclasses import asdict
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

from ..models import (
    AnalysisResult, ArrangedSection, RenderParams,
    ScoreBreakdown, Segment, SongPlan, Structure,
)
from ..renderer import render
from ..arranger import arrange
from ..structures import load_structure


class GUIState:
    """Mutable state shared across request handlers."""

    def __init__(
        self,
        analysis: AnalysisResult,
        segments: list[Segment],
        dist_matrix: np.ndarray,
        plan: SongPlan,
        output_path: str,
    ):
        self.analysis = analysis
        self.segments = segments
        self.dist_matrix = dist_matrix
        self.plan = plan
        self.output_path = output_path
        self.lock = threading.Lock()


def _segment_to_dict(seg: Segment) -> dict:
    return {
        "index": seg.index,
        "source_start": round(seg.source_start, 3),
        "source_end": round(seg.source_end, 3),
        "duration": round(seg.duration, 3),
        "mean_energy": round(seg.mean_energy, 4),
        "mean_brightness": round(seg.mean_brightness, 4),
        "onset_density": round(seg.onset_density, 4),
        "energy_tier": seg.energy_tier,
        "trend": seg.trend,
    }


def _breakdown_to_dict(bd: ScoreBreakdown) -> dict:
    return {
        "energy_fit": round(bd.energy_fit, 4),
        "variety": round(bd.variety, 4),
        "duration_fit": round(bd.duration_fit, 4),
        "temporal_fit": round(bd.temporal_fit, 4),
        "transition_score": round(bd.transition_score, 4),
    }


def _build_state_json(state: GUIState) -> dict:
    plan = state.plan
    info = plan.audio_info

    sections = []
    for arr in plan.arranged_sections:
        candidates = []
        for seg_idx, score, bd in state.plan.candidates_per_role.get(arr.role, []):
            seg = state.segments[seg_idx] if seg_idx < len(state.segments) else None
            if seg:
                candidates.append({
                    "segment": _segment_to_dict(seg),
                    "score": round(score, 4),
                    "score_breakdown": _breakdown_to_dict(bd),
                })

        sections.append({
            "role": arr.role,
            "segment": _segment_to_dict(arr.segment),
            "target_duration": round(arr.target_duration, 3),
            "actual_duration": round(arr.actual_duration, 3),
            "score": round(arr.score, 4),
            "score_breakdown": _breakdown_to_dict(arr.score_breakdown),
            "candidates": candidates,
        })

    return {
        "audio_info": {
            "path": info.path,
            "duration": round(info.duration, 3),
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "tempo_bpm": round(info.tempo_bpm, 2),
        },
        "structure": {
            "name": plan.structure.name,
            "description": plan.structure.description,
            "sections": [
                {"role": s.role, "energy": s.energy,
                 "relative_duration": s.relative_duration,
                 "similar_to": s.similar_to}
                for s in plan.structure.sections
            ],
        },
        "target_duration": plan.target_duration,
        "render_params": {
            "crossfade": plan.render_params.crossfade,
            "fade_in": plan.render_params.fade_in,
            "fade_out": plan.render_params.fade_out,
        },
        "output_path": state.output_path,
        "sections": sections,
        "segments": [_segment_to_dict(s) for s in state.segments],
    }


class GUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the GUI API."""

    state: GUIState  # set via partial

    def log_message(self, format, *args):
        pass  # suppress default logging

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _parse_qs(self) -> dict:
        parsed = urlparse(self.path)
        return {k: v[0] for k, v in parse_qs(parsed.query).items()}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path == "/api/state":
            self._handle_state()
        elif path == "/api/audio":
            self._handle_audio()
        elif path == "/api/waveform":
            self._handle_waveform()
        elif path == "/api/rendered":
            self._handle_rendered()
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/update":
            self._handle_update()
        elif parsed.path == "/api/render":
            self._handle_render()
        elif parsed.path == "/api/load-edl":
            self._handle_load_edl()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_html(self):
        html_path = Path(__file__).parent / "index.html"
        content = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_state(self):
        with self.state.lock:
            data = _build_state_json(self.state)
        self._send_json(data)

    def _handle_audio(self):
        params = self._parse_qs()
        try:
            start = float(params["start"])
            end = float(params["end"])
        except (KeyError, ValueError):
            self.send_error(400, "start and end params required")
            return

        y = np.asarray(self.state.analysis.y_render)
        sr = self.state.analysis.sr_render
        s_sample = max(0, int(start * sr))
        e_sample = min(y.shape[-1], int(end * sr))
        chunk = y[..., s_sample:e_sample]

        buf = io.BytesIO()
        data = chunk.T if chunk.ndim == 2 else chunk
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        self._send_bytes(buf.getvalue(), "audio/wav")

    def _handle_waveform(self):
        params = self._parse_qs()
        try:
            start = float(params["start"])
            end = float(params["end"])
            width = int(params.get("width", "200"))
        except (KeyError, ValueError):
            self.send_error(400, "start and end params required")
            return

        y = np.asarray(self.state.analysis.y_render)
        sr = self.state.analysis.sr_render
        # Use mono for waveform
        if y.ndim == 2:
            y_mono = y.mean(axis=0)
        else:
            y_mono = y
        s_sample = max(0, int(start * sr))
        e_sample = min(len(y_mono), int(end * sr))
        chunk = y_mono[s_sample:e_sample]

        if len(chunk) == 0:
            self._send_json({"peaks": []})
            return

        # Downsample to `width` peaks
        width = max(1, min(width, len(chunk)))
        bucket_size = len(chunk) // width
        peaks = []
        for i in range(width):
            bucket = chunk[i * bucket_size:(i + 1) * bucket_size]
            peaks.append(round(float(np.max(np.abs(bucket))), 4))

        self._send_json({"peaks": peaks})

    def _handle_rendered(self):
        path = self.state.output_path
        if not os.path.exists(path):
            self.send_error(404, "Rendered file not found")
            return
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        mime = {
            ".opus": "audio/opus",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }.get(ext, "application/octet-stream")
        self._send_bytes(data, mime)

    def _handle_update(self):
        """Update the arrangement (swap segments, reorder, adjust durations)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        with self.state.lock:
            plan = self.state.plan
            segments = self.state.segments
            new_sections = body.get("sections", [])

            if body.get("output_path"):
                self.state.output_path = body["output_path"]

            # Rebuild arranged_sections from the update
            updated = []
            for section_data in new_sections:
                role = section_data["role"]
                seg_idx = section_data["segment_index"]
                actual_dur = section_data.get("actual_duration")

                seg = segments[seg_idx]

                # Find original arranged section for this role to get target_duration
                orig = next(
                    (a for a in plan.arranged_sections if a.role == role), None
                )
                target_dur = orig.target_duration if orig else actual_dur or seg.duration

                if actual_dur is None:
                    actual_dur = min(seg.duration, target_dur)

                # Find score/breakdown from candidates
                score = 0.0
                breakdown = ScoreBreakdown(energy_fit=0, variety=0, duration_fit=0)
                for c_idx, c_score, c_bd in plan.candidates_per_role.get(role, []):
                    if c_idx == seg_idx:
                        score = c_score
                        breakdown = c_bd
                        break

                updated.append(ArrangedSection(
                    role=role,
                    segment=seg,
                    target_duration=target_dur,
                    actual_duration=actual_dur,
                    score=score,
                    score_breakdown=breakdown,
                ))

            plan.arranged_sections = updated
            data = _build_state_json(self.state)

        self._send_json(data)

    def _handle_render(self):
        """Re-render the current arrangement."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        with self.state.lock:
            if body.get("output_path"):
                self.state.output_path = body["output_path"]

            plan = self.state.plan
            plan.render_params.output_path = self.state.output_path

            output_duration = render(plan, self.state.analysis, self.state.output_path)
            data = _build_state_json(self.state)
            data["output_duration"] = round(output_duration, 3)

        self._send_json(data)

    def _handle_load_edl(self):
        """Load an EDL and rebuild the arrangement to match it.

        Accepts either:
          {"edl": { ... }}         — inline EDL content (from browser file picker)
          {"path": "/path/to/x.edl.json"}  — server-side file path
        """
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        edl = body.get("edl")
        if not edl:
            edl_path = body.get("path")
            if not edl_path or not os.path.exists(edl_path):
                self._send_json({"error": "EDL file not found"}, status=400)
                return
            try:
                with open(edl_path, encoding="utf-8") as f:
                    edl = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                self._send_json({"error": f"Cannot read EDL: {exc}"}, status=400)
                return

        with self.state.lock:
            try:
                self._apply_edl(edl)
                edl_output = edl.get("render", {}).get("output_path", "")
                if edl_output:
                    self.state.output_path = edl_output
                data = _build_state_json(self.state)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
                return

        self._send_json(data)

    def _apply_edl(self, edl: dict):
        """Rebuild state.plan to match an EDL's sections."""
        segments = self.state.segments

        # Load the structure named in the EDL and re-arrange to get candidates
        structure_name = edl.get("structure", {}).get("name")
        if not structure_name:
            raise ValueError("EDL missing structure.name")

        structure = load_structure(structure_name)
        target_duration = edl.get("structure", {}).get(
            "target_duration", self.state.plan.target_duration
        )

        # Rebuild render params from EDL
        edl_render = edl.get("render", {})
        render_params = RenderParams(
            crossfade=edl_render.get("crossfade", self.state.plan.render_params.crossfade),
            fade_in=edl_render.get("fade_in", self.state.plan.render_params.fade_in),
            fade_out=edl_render.get("fade_out", self.state.plan.render_params.fade_out),
            output_path=edl_render.get("output_path", self.state.output_path),
        )

        # Run arrange() to populate candidates_per_role
        new_plan = arrange(
            segments,
            self.state.dist_matrix,
            structure,
            self.state.analysis.audio_info,
            target_duration,
            render_params,
            beat_times=self.state.analysis.beat_times,
        )

        # Now override arranged_sections with the EDL's actual assignments
        edl_sections = edl.get("sections", [])
        overridden = []
        for edl_sec in edl_sections:
            role = edl_sec["role"]
            source_start = edl_sec["source_start"]
            actual_dur = edl_sec["duration"]

            # Match to a segment by closest source_start
            best_seg = min(segments, key=lambda s: abs(s.source_start - source_start))
            if abs(best_seg.source_start - source_start) > 1.0:
                raise ValueError(
                    f"No segment near source_start={source_start} for role '{role}'"
                )

            # Get target_duration from the new plan if available
            plan_sec = next(
                (a for a in new_plan.arranged_sections if a.role == role), None
            )
            target_dur = plan_sec.target_duration if plan_sec else actual_dur

            # Look up score from candidates
            score = edl_sec.get("score", 0.0)
            bd_raw = edl_sec.get("score_breakdown", {})
            breakdown = ScoreBreakdown(
                energy_fit=bd_raw.get("energy_fit", 0),
                variety=bd_raw.get("variety", 0),
                duration_fit=bd_raw.get("duration_fit", 0),
                temporal_fit=bd_raw.get("temporal_fit", 1.0),
                transition_score=bd_raw.get("transition_score", 1.0),
            )

            overridden.append(ArrangedSection(
                role=role,
                segment=best_seg,
                target_duration=target_dur,
                actual_duration=actual_dur,
                score=score,
                score_breakdown=breakdown,
            ))

        new_plan.arranged_sections = overridden
        self.state.plan = new_plan


def start_gui(
    analysis: AnalysisResult,
    segments: list[Segment],
    dist_matrix: np.ndarray,
    plan: SongPlan,
    output_path: str,
    port: int = 8765,
) -> None:
    """Start the GUI server and open the browser."""
    state = GUIState(analysis, segments, dist_matrix, plan, output_path)

    handler = type("Handler", (GUIHandler,), {"state": state})
    server = HTTPServer(("127.0.0.1", port), handler)

    url = f"http://localhost:{port}"
    print(f"GUI server running at {url}")
    print("Press Ctrl+C to stop.")

    # Open browser after a short delay
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
