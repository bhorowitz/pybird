"""Bridge from `lya_reduction_sympy.py` outputs into the lightweight pipeline.

This module lets the current descriptor generator reuse the real SymPy-based
13-sector reductions where available, while keeping fallback paths for channels
that remain intentionally outside the shifted-J descriptor framework (notably
the standard F3/G3 channels handled by existing CLASS-PT radial-kernel code).
"""

from __future__ import annotations

import json
from pathlib import Path

from lya_reduction_sympy import generate_13_descriptors


ROOT = Path(__file__).resolve().parent
SYMPY_OUTDIR = ROOT / "lya_sympy_descriptors"


def generate_sympy_13_descriptor_dicts(channel_name: str) -> list[dict]:
    """Generate and load SymPy-backed 13-sector descriptors for one channel."""
    SYMPY_OUTDIR.mkdir(parents=True, exist_ok=True)
    descriptors = generate_13_descriptors(channel_name, outdir=str(SYMPY_OUTDIR))
    if not descriptors:
        return []

    loaded = []
    for descriptor in descriptors:
        mu_power = descriptor["mu_power"]
        path = SYMPY_OUTDIR / f"LYA_{channel_name}__MU{mu_power}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("metadata", {})
        payload["metadata"]["status"] = "sympy-reduced"
        payload["metadata"]["reduction_source"] = "lya_reduction_sympy.generate_13_descriptors"
        loaded.append(payload)
    return loaded
