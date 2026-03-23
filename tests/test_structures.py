import json
import pytest
from pathlib import Path
import tempfile

from jam2song.structures import load_structure, list_structures
from jam2song.models import Structure, SectionSpec


# --- Built-in loading ---

def test_load_loop_build_drop():
    s = load_structure("loop_build_drop")
    assert isinstance(s, Structure)
    assert s.name == "loop_build_drop"
    assert len(s.sections) == 8  # intro, intro_fill, build_1, drop_1, breakdown, build_2, drop_2, outro


def test_load_verse_chorus():
    s = load_structure("verse_chorus")
    assert s.name == "verse_chorus"
    assert len(s.sections) == 9  # intro, intro_fill, verse_1, chorus_1, verse_2, chorus_2, bridge, chorus_3, outro
    # chorus_2 and chorus_3 reference chorus_1 (earlier)
    chorus_2 = next(sec for sec in s.sections if sec.role == "chorus_2")
    assert chorus_2.similar_to == "chorus_1"


def test_load_highlight_reel():
    s = load_structure("highlight_reel")
    assert s.name == "highlight_reel"
    assert len(s.sections) == 6
    last = s.sections[-1]
    assert last.energy == "falling"


def test_load_condensed_jam():
    s = load_structure("condensed_jam")
    assert s.name == "condensed_jam"
    assert len(s.sections) == 18
    # groove_2 references groove_1, peak_2 references peak_1
    groove_2 = next(sec for sec in s.sections if sec.role == "groove_2")
    assert groove_2.similar_to == "groove_1"
    peak_2 = next(sec for sec in s.sections if sec.role == "peak_2")
    assert peak_2.similar_to == "peak_1"
    # climax has the largest relative_duration
    climax = next(sec for sec in s.sections if sec.role == "climax")
    assert climax.relative_duration == max(sec.relative_duration for sec in s.sections)


def test_load_slow_burn():
    s = load_structure("slow_burn")
    assert s.name == "slow_burn"
    assert len(s.sections) == 8
    # Energy only goes up (no falling/low after a high) until outro
    climax = next(sec for sec in s.sections if sec.role == "climax")
    assert climax.similar_to == "peak"
    assert climax.relative_duration == max(sec.relative_duration for sec in s.sections)


def test_load_ambient_drift():
    s = load_structure("ambient_drift")
    assert s.name == "ambient_drift"
    assert len(s.sections) == 8
    # No high energy sections
    energies = {sec.energy for sec in s.sections}
    assert "high" not in energies
    # similar_to callbacks
    texture_2 = next(sec for sec in s.sections if sec.role == "texture_2")
    assert texture_2.similar_to == "texture_1"
    drift_2 = next(sec for sec in s.sections if sec.role == "drift_2")
    assert drift_2.similar_to == "drift_1"


def test_load_call_and_response():
    s = load_structure("call_and_response")
    assert s.name == "call_and_response"
    assert len(s.sections) == 8
    # Alternating pattern with callbacks
    call_2 = next(sec for sec in s.sections if sec.role == "call_2")
    assert call_2.similar_to == "call_1"
    response_2 = next(sec for sec in s.sections if sec.role == "response_2")
    assert response_2.similar_to == "response_1"


def test_load_micro_edit():
    s = load_structure("micro_edit")
    assert s.name == "micro_edit"
    assert len(s.sections) == 4
    # No intro/outro anchoring — all energy-based roles
    roles = [sec.role for sec in s.sections]
    assert not any(r.startswith("intro") or r.startswith("outro") for r in roles)


def test_list_structures_contains_builtins():
    names = list_structures()
    assert "loop_build_drop" in names
    assert "verse_chorus" in names
    assert "highlight_reel" in names
    assert "condensed_jam" in names
    assert "slow_burn" in names
    assert "ambient_drift" in names
    assert "call_and_response" in names
    assert "micro_edit" in names


# --- Custom JSON from path ---

def test_load_custom_json(tmp_path):
    data = {
        "name": "my_custom",
        "description": "test",
        "sections": [
            {"role": "a", "energy": "low", "relative_duration": 1},
            {"role": "b", "energy": "high", "relative_duration": 2},
        ],
    }
    p = tmp_path / "my_custom.json"
    p.write_text(json.dumps(data))
    s = load_structure(str(p))
    assert s.name == "my_custom"
    assert len(s.sections) == 2


def test_load_missing_structure_raises():
    with pytest.raises(FileNotFoundError):
        load_structure("nonexistent_structure_xyz")


# --- Validation failures ---

def _load_from_dict(data):
    from jam2song.structures import _validate_and_build
    return _validate_and_build(data, "<test>")


def test_missing_name():
    with pytest.raises(ValueError, match="name"):
        _load_from_dict({"sections": [{"role": "a", "energy": "low", "relative_duration": 1}]})


def test_empty_name():
    with pytest.raises(ValueError, match="name"):
        _load_from_dict({"name": "  ", "sections": [{"role": "a", "energy": "low", "relative_duration": 1}]})


def test_missing_sections():
    with pytest.raises(ValueError, match="sections"):
        _load_from_dict({"name": "x"})


def test_empty_sections():
    with pytest.raises(ValueError, match="sections"):
        _load_from_dict({"name": "x", "sections": []})


def test_invalid_energy():
    with pytest.raises(ValueError, match="energy"):
        _load_from_dict({
            "name": "x",
            "sections": [{"role": "a", "energy": "loud", "relative_duration": 1}],
        })


def test_negative_duration():
    with pytest.raises(ValueError, match="relative_duration"):
        _load_from_dict({
            "name": "x",
            "sections": [{"role": "a", "energy": "low", "relative_duration": -1}],
        })


def test_zero_duration():
    with pytest.raises(ValueError, match="relative_duration"):
        _load_from_dict({
            "name": "x",
            "sections": [{"role": "a", "energy": "low", "relative_duration": 0}],
        })


def test_duplicate_roles():
    with pytest.raises(ValueError, match="duplicate"):
        _load_from_dict({
            "name": "x",
            "sections": [
                {"role": "a", "energy": "low", "relative_duration": 1},
                {"role": "a", "energy": "high", "relative_duration": 1},
            ],
        })


def test_similar_to_forward_reference():
    with pytest.raises(ValueError, match="similar_to"):
        _load_from_dict({
            "name": "x",
            "sections": [
                {"role": "a", "energy": "low", "relative_duration": 1, "similar_to": "b"},
                {"role": "b", "energy": "high", "relative_duration": 1},
            ],
        })


def test_similar_to_self_reference():
    with pytest.raises(ValueError, match="similar_to"):
        _load_from_dict({
            "name": "x",
            "sections": [
                {"role": "a", "energy": "low", "relative_duration": 1, "similar_to": "a"},
            ],
        })


def test_single_section_is_valid():
    s = _load_from_dict({
        "name": "minimal",
        "sections": [{"role": "only", "energy": "mid", "relative_duration": 1}],
    })
    assert len(s.sections) == 1


def test_all_valid_energies():
    sections = [
        {"role": e, "energy": e, "relative_duration": 1}
        for e in ["low", "mid", "high", "rising", "falling"]
    ]
    s = _load_from_dict({"name": "all_energies", "sections": sections})
    assert len(s.sections) == 5


def test_similar_to_valid_earlier_role():
    s = _load_from_dict({
        "name": "x",
        "sections": [
            {"role": "first", "energy": "high", "relative_duration": 2},
            {"role": "second", "energy": "high", "relative_duration": 2, "similar_to": "first"},
        ],
    })
    assert s.sections[1].similar_to == "first"
