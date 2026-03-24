"""Helpers for LyA matrix descriptor JSON files.

The immediate goal is not full symbolic reduction yet. This file freezes the
descriptor schema from `lya_matrix_generation_guide.txt` and provides builders
and validators so the upcoming symbolic layer has a concrete target format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


ALLOWED_13_MU_POWERS = (0, 2, 4, 6)
ALLOWED_22_MU_POWERS = (0, 2, 4, 6, 8)
ALLOWED_MASTER_COMPONENTS = (
    "J",
    "A5", "B5", "C5",
    "A6", "B6", "C6", "D6",
    "A7", "B7", "C7", "D7",
    "A8", "B8", "C8", "D8", "E8",
)


@dataclass(frozen=True)
class DescriptorTerm:
    coeff_rational: dict[str, int] | None
    coeff_float: float | None
    los_rank_r: int
    delta_nu1: int
    delta_nu2: int
    master_component: str
    extra_k_power: int = 0
    extra_mu_power: int = 0
    numerator_tag: str = ""
    notes: str = ""


@dataclass(frozen=True)
class DescriptorSymmetry:
    type: str = "none"
    exchange_q_kmq: bool = False


@dataclass(frozen=True)
class ChannelDescriptor:
    version: str
    sector: str
    channel_name: str
    channel_id: int
    mu_power: int
    k_power_prefactor: int
    radial_rank: int
    symmetry: DescriptorSymmetry = field(default_factory=DescriptorSymmetry)
    master_family: str = "LOS_SHIFTED_J"
    terms: tuple[DescriptorTerm, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


def validate_descriptor(descriptor: ChannelDescriptor) -> None:
    if descriptor.sector not in {"13", "22"}:
        raise ValueError(f"Unsupported sector {descriptor.sector!r}")
    allowed_mu = ALLOWED_13_MU_POWERS if descriptor.sector == "13" else ALLOWED_22_MU_POWERS
    if descriptor.mu_power not in allowed_mu:
        raise ValueError(f"mu_power={descriptor.mu_power} is not allowed for sector {descriptor.sector}")
    for term in descriptor.terms:
        if term.los_rank_r < 0 or term.los_rank_r > 8:
            raise ValueError(f"LOS rank must be in [0,8], got {term.los_rank_r}")
        if term.master_component not in ALLOWED_MASTER_COMPONENTS:
            raise ValueError(f"Unsupported master component {term.master_component!r}")
        if term.coeff_rational is None and term.coeff_float is None:
            raise ValueError("Each term must define coeff_rational or coeff_float")


def descriptor_to_dict(descriptor: ChannelDescriptor) -> dict[str, object]:
    validate_descriptor(descriptor)
    data = asdict(descriptor)
    data["terms"] = [asdict(term) for term in descriptor.terms]
    data["symmetry"] = asdict(descriptor.symmetry)
    return data


def write_descriptor(descriptor: ChannelDescriptor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(descriptor_to_dict(descriptor), indent=2) + "\n", encoding="utf-8")
