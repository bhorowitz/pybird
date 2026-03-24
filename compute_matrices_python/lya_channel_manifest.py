"""Manifest helpers for mixed-source LyA channel production.

This tracks whether a given LyA channel/mu-power pair comes from:
- a descriptor-generated shifted-J reduction,
- the existing CLASS-PT backend-import F3/G3 path,
- or a placeholder still awaiting implementation.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_manifest(descriptor_dir: Path) -> dict[str, object]:
    entries = []
    for path in sorted(descriptor_dir.glob("LYA_T13__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        entries.append(
            {
                "channel_name": payload["channel_name"],
                "mu_power": payload["mu_power"],
                "status": metadata.get("status", "unknown"),
                "reduction_source": metadata.get("reduction_source"),
                "import_source": metadata.get("import_source"),
                "n_terms": len(payload.get("terms", [])),
                "path": str(path),
            }
        )

    summary = {
        "total_entries": len(entries),
        "backend_import_entries": sum(1 for e in entries if e["status"] == "backend-import"),
        "descriptor_generated_entries": sum(1 for e in entries if e["status"] in {"sympy-reduced", "reducer-prototype"}),
        "physically_zero_entries": sum(1 for e in entries if e["status"] == "physically-zero"),
        "placeholder_entries": sum(1 for e in entries if e["status"] == "placeholder"),
        "radial_1d_entries": sum(1 for e in entries if e["status"] == "radial-1d"),
        "entries": entries,
    }
    return summary


def write_manifest(descriptor_dir: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_manifest(descriptor_dir), indent=2) + "\n", encoding="utf-8")
    return output_path
