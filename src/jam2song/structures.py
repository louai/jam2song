import json
from pathlib import Path

from .models import SectionSpec, Structure

STRUCTURES_DIR = Path(__file__).parent / "structures"
VALID_ENERGIES = {"low", "mid", "high", "rising", "falling"}


def list_structures() -> list[str]:
    return sorted(p.stem for p in STRUCTURES_DIR.glob("*.json"))


def load_structure(name_or_path: str) -> Structure:
    path = Path(name_or_path)
    if not path.exists():
        path = STRUCTURES_DIR / f"{name_or_path}.json"
    if not path.exists():
        raise FileNotFoundError(f"Structure not found: {name_or_path!r}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return _validate_and_build(data, str(path))


def _validate_and_build(data: dict, source: str) -> Structure:
    # Rule 1: name is a non-empty string
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise ValueError(f"[{source}] 'name' must be a non-empty string")

    # Rule 2: sections is a non-empty array
    sections_raw = data.get("sections")
    if not isinstance(sections_raw, list) or len(sections_raw) == 0:
        raise ValueError(f"[{source}] 'sections' must be a non-empty array")

    seen_roles: set[str] = set()
    sections: list[SectionSpec] = []

    for i, s in enumerate(sections_raw):
        loc = f"[{source}] sections[{i}]"

        # Rule 3a: role present and non-empty string
        if not isinstance(s.get("role"), str) or not s["role"].strip():
            raise ValueError(f"{loc} 'role' must be a non-empty string")

        # Rule 3b: energy is valid
        if s.get("energy") not in VALID_ENERGIES:
            raise ValueError(
                f"{loc} 'energy' must be one of {sorted(VALID_ENERGIES)}, got {s.get('energy')!r}"
            )

        # Rule 3c: relative_duration is a positive number
        rd = s.get("relative_duration")
        if not isinstance(rd, (int, float)) or rd <= 0:
            raise ValueError(f"{loc} 'relative_duration' must be a positive number")

        # Rule 4: roles must be unique
        if s["role"] in seen_roles:
            raise ValueError(f"{loc} duplicate role {s['role']!r}")

        # Rule 5: similar_to must reference an earlier role (not self, not forward)
        sim = s.get("similar_to")
        if sim is not None:
            if sim not in seen_roles:
                raise ValueError(
                    f"{loc} 'similar_to' must reference an earlier role, got {sim!r}"
                )

        seen_roles.add(s["role"])
        sections.append(
            SectionSpec(
                role=s["role"],
                energy=s["energy"],
                relative_duration=float(rd),
                similar_to=sim,
            )
        )

    return Structure(
        name=data["name"],
        description=data.get("description", ""),
        sections=sections,
    )
