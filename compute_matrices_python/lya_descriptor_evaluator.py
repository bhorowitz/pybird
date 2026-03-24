"""Tiny prototype evaluator for LyA descriptors.

This is intentionally a minimal development utility. It evaluates the sparse
term list symbolically/numerically for a given `(nu1, nu2, mu, k)` point using
small placeholder master-component implementations. This validates the JSON
descriptor contract before the full matrix generator is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from mpmath import gamma, pi


def J(nu1, nu2):
    return (
        gamma(1.5 - nu1)
        * gamma(1.5 - nu2)
        * gamma(nu1 + nu2 - 1.5)
        / (gamma(nu1) * gamma(nu2) * gamma(3.0 - nu2 - nu1))
    ) / (8.0 * pi ** 1.5)


def _master_component_value(component: str, nu1, nu2):
    base = J(nu1, nu2)
    if component == "J":
        return base
    if component == "A1":
        return (nu1 + nu2) * base
    if component == "A2":
        return (nu1 + nu2 + 1) * base
    if component == "B2":
        return (nu1 - nu2) * base
    if component == "A3":
        return (nu1 + nu2) * (nu1 + nu2 + 1) * base
    if component == "B3":
        return (nu1 - nu2) * (nu1 + nu2) * base
    if component == "A4":
        return (nu1 + nu2 + 2) * (nu1 + nu2) * base
    if component == "B4":
        return (nu1 - nu2) * (nu1 + nu2 + 1) * base
    if component == "C4":
        return (nu1 - nu2) ** 2 * base
    if component == "A5":
        return (nu1 + nu2 + 3) * (nu1 + nu2) * base
    if component == "A6":
        return (nu1 + nu2) * base
    if component == "B5":
        return (nu1 - nu2) * (nu1 + nu2 + 2) * base
    if component == "C5":
        return (nu1 - nu2) ** 2 * (nu1 + nu2) * base
    if component == "B6":
        return (nu1 - nu2) * (nu1 + nu2 + 3) * base
    if component == "C6":
        return (nu1 - nu2) ** 2 * (nu1 + nu2 + 1) * base
    if component == "D6":
        return (nu1 - nu2) ** 3 * base
    raise ValueError(f"Prototype evaluator does not support master component {component!r}")


def evaluate_descriptor_dict(descriptor: dict, nu1, nu2, mu, k):
    value = 0.0
    channel_prefactor = (k ** descriptor["k_power_prefactor"]) * (mu ** descriptor["mu_power"])
    for term in descriptor["terms"]:
        coeff = 0.0
        if term["coeff_rational"] is not None:
            coeff = term["coeff_rational"]["num"] / term["coeff_rational"]["den"]
        elif term["coeff_float"] is not None:
            coeff = term["coeff_float"]
        master = _master_component_value(
            term["master_component"],
            nu1 - term["delta_nu1"],
            nu2 - term["delta_nu2"],
        )
        value += coeff * master * (k ** term["extra_k_power"]) * (mu ** term["extra_mu_power"])
    return channel_prefactor * value


def load_descriptor(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
