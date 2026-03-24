"""Production runtime evaluation for LyA descriptors.

This module evaluates a descriptor at complex FFTLog mode-space points using
the dedicated master-library implementation in `lya_master_library.py`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from lya_master_library import COMPONENT_SHIFTS, J, ShiftedJCache, eval_master_component, eval_master_component_cached


def load_descriptor(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_descriptor_dict_runtime(descriptor: dict, nu1, nu2, mu, k):
    value = 0.0j
    channel_prefactor = (k ** descriptor["k_power_prefactor"]) * (mu ** descriptor["mu_power"])
    for term in descriptor["terms"]:
        coeff = 0.0
        if term["coeff_rational"] is not None:
            coeff = term["coeff_rational"]["num"] / term["coeff_rational"]["den"]
        elif term["coeff_float"] is not None:
            coeff = term["coeff_float"]

        master = eval_master_component(
            term["master_component"],
            nu1 + term["delta_nu1"],
            nu2 + term["delta_nu2"],
        )
        value += coeff * master * (k ** term["extra_k_power"]) * (mu ** term["extra_mu_power"])
    return channel_prefactor * value


def evaluate_descriptor_physical(descriptor: dict, nu1, nu2, mu, k):
    """Evaluate the full physical descriptor coefficient at a chosen point."""
    return evaluate_descriptor_dict_runtime(descriptor, nu1=nu1, nu2=nu2, mu=mu, k=k)


def descriptor_k_powers(descriptor: dict) -> list[int]:
    """Return the sorted unique per-term extra k-power sectors present in a descriptor."""
    return sorted({int(term["extra_k_power"]) for term in descriptor["terms"]})


def split_descriptor_by_k_power(descriptor: dict) -> dict[int, dict]:
    """Split a descriptor into fixed-k-power groups and promote that power to descriptor metadata.

    Each returned descriptor clone contains only terms from one extra_k_power sector,
    normalized so the per-term `extra_k_power` is zero. The promoted power is stored in
    `k_power` and added to the descriptor-level `k_power_prefactor`.
    """
    grouped: dict[int, list[dict]] = {}
    for term in descriptor["terms"]:
        k_power = int(term["extra_k_power"])
        grouped.setdefault(k_power, []).append(term)

    out: dict[int, dict] = {}
    for k_power, terms in grouped.items():
        clone = dict(descriptor)
        metadata = dict(descriptor.get("metadata", {}))
        metadata["split_k_power"] = k_power
        clone["metadata"] = metadata
        clone["k_power"] = k_power
        clone["k_power_prefactor"] = int(descriptor.get("k_power_prefactor", 0)) + k_power
        normalized_terms = []
        for term in terms:
            new_term = dict(term)
            new_term["extra_k_power"] = 0
            normalized_terms.append(new_term)
        clone["terms"] = normalized_terms
        out[k_power] = clone
    return out


def evaluate_descriptor_kernel(descriptor: dict, nu1, nu2):
    """Evaluate a pure nu-space kernel descriptor with no physical mu/k applied.

    This is only valid after splitting to a fixed-k-power sector and normalizing all
    term `extra_k_power` values to zero.
    """
    if any(int(term["extra_k_power"]) != 0 for term in descriptor["terms"]):
        raise ValueError("Kernel evaluation requires a fixed-k-power descriptor with normalized term extra_k_power=0")
    if int(descriptor.get("mu_power", 0)) != 0 and any(int(term.get("extra_mu_power", 0)) != 0 for term in descriptor["terms"]):
        raise ValueError("Kernel evaluation requires mu dependence to be carried only by file identity, not runtime factors")

    value = 0.0j
    for term in descriptor["terms"]:
        coeff = 0.0
        if term["coeff_rational"] is not None:
            coeff = term["coeff_rational"]["num"] / term["coeff_rational"]["den"]
        elif term["coeff_float"] is not None:
            coeff = term["coeff_float"]

        master = eval_master_component(
            term["master_component"],
            nu1 + term["delta_nu1"],
            nu2 + term["delta_nu2"],
        )
        value += coeff * master
    return value


def evaluate_descriptor_dict_runtime_cached(descriptor: dict, nu1, nu2, mu, k, *, return_stats: bool = False):
    value = 0.0j
    channel_prefactor = (k ** descriptor["k_power_prefactor"]) * (mu ** descriptor["mu_power"])
    jcache = ShiftedJCache(nu1, nu2)

    for term in descriptor["terms"]:
        coeff = 0.0
        if term["coeff_rational"] is not None:
            coeff = term["coeff_rational"]["num"] / term["coeff_rational"]["den"]
        elif term["coeff_float"] is not None:
            coeff = term["coeff_float"]

        delta_nu1 = int(term["delta_nu1"])
        delta_nu2 = int(term["delta_nu2"])

        master = eval_master_component_cached(
            term["master_component"],
            jcache.with_offset(delta_nu1, delta_nu2),
        )

        value += coeff * master * (k ** term["extra_k_power"]) * (mu ** term["extra_mu_power"])

    result = channel_prefactor * value
    if return_stats:
        return result, jcache.stats()
    return result


def eval_master_component_from_metadata(component: str, nu1, nu2):
    if component not in COMPONENT_SHIFTS:
        return eval_master_component(component, nu1, nu2)

    total = 0.0j
    for coeff, shift1, shift2 in COMPONENT_SHIFTS[component]:
        total += coeff * J(complex(nu1) + shift1, complex(nu2) + shift2)
    return total


def collect_required_shifts(descriptors: list[dict]) -> list[tuple[int, int]]:
    shifts: set[tuple[int, int]] = set()
    for descriptor in descriptors:
        for term in descriptor["terms"]:
            component_terms = COMPONENT_SHIFTS.get(term["master_component"])
            if component_terms is None:
                continue
            delta_nu1 = int(term["delta_nu1"])
            delta_nu2 = int(term["delta_nu2"])
            for _, shift1, shift2 in component_terms:
                shifts.add((delta_nu1 + shift1, delta_nu2 + shift2))
    return sorted(shifts)


def flatten_descriptor(descriptor: dict) -> dict[tuple[int, int, int, int], complex]:
    grouped: defaultdict[tuple[int, int, int, int], complex] = defaultdict(complex)
    base_k = descriptor["k_power_prefactor"]
    base_mu = descriptor["mu_power"]

    for term in descriptor["terms"]:
        coeff = 0.0
        if term["coeff_rational"] is not None:
            coeff = term["coeff_rational"]["num"] / term["coeff_rational"]["den"]
        elif term["coeff_float"] is not None:
            coeff = term["coeff_float"]

        component_terms = COMPONENT_SHIFTS.get(term["master_component"])
        if component_terms is None:
            raise ValueError(f"No COMPONENT_SHIFTS metadata for component {term['master_component']!r}")

        delta_nu1 = int(term["delta_nu1"])
        delta_nu2 = int(term["delta_nu2"])
        total_k_power = int(base_k + term["extra_k_power"])
        total_mu_power = int(base_mu + term["extra_mu_power"])

        for local_coeff, shift1, shift2 in component_terms:
            grouped[(delta_nu1 + shift1, delta_nu2 + shift2, total_k_power, total_mu_power)] += coeff * local_coeff

    return dict(grouped)


def prepare_flattened_descriptor(descriptor: dict) -> dict:
    grouped = flatten_descriptor(descriptor)
    unique_shift_keys = tuple(sorted({(shift1, shift2) for shift1, shift2, _, _ in grouped}))
    shift_index = {key: index for index, key in enumerate(unique_shift_keys)}
    atoms = tuple(
        (shift_index[(shift1, shift2)], total_k_power, total_mu_power, coeff)
        for (shift1, shift2, total_k_power, total_mu_power), coeff in grouped.items()
    )
    return {
        "grouped": grouped,
        "unique_shift_keys": unique_shift_keys,
        "atoms": atoms,
        "unique_grouped_atoms": len(grouped),
    }


def evaluate_flattened_descriptor(grouped: dict[tuple[int, int, int, int], complex], nu1, nu2, mu, k, *, return_stats: bool = False):
    jcache = ShiftedJCache(nu1, nu2)
    total = 0.0j
    contribution_abs_sum = 0.0
    largest_abs_contribution = 0.0

    unique_shift_keys = {(shift1, shift2) for shift1, shift2, _, _ in grouped}
    local_j = {key: jcache.get(*key) for key in unique_shift_keys}

    for (shift1, shift2, total_k_power, total_mu_power), coeff in grouped.items():
        contribution = coeff * local_j[(shift1, shift2)] * (k ** total_k_power) * (mu ** total_mu_power)
        total += contribution
        abs_contribution = abs(contribution)
        contribution_abs_sum += abs_contribution
        largest_abs_contribution = max(largest_abs_contribution, abs_contribution)

    if not return_stats:
        return total

    condition_proxy = float("inf") if abs(total) == 0.0 and contribution_abs_sum > 0.0 else contribution_abs_sum / max(abs(total), 1e-300)
    return total, {
        **jcache.stats(),
        "unique_grouped_atoms": len(grouped),
        "largest_abs_contribution": largest_abs_contribution,
        "condition_proxy": condition_proxy,
    }


def evaluate_preflattened_descriptor(prepared: dict, nu1, nu2, mu, k, *, return_stats: bool = False):
    unique_shift_keys = prepared["unique_shift_keys"]
    atoms = prepared["atoms"]

    jcache = ShiftedJCache(nu1, nu2)
    total = 0.0j
    contribution_abs_sum = 0.0
    largest_abs_contribution = 0.0

    local_j = tuple(jcache.get(*key) for key in unique_shift_keys)

    for shift_index, total_k_power, total_mu_power, coeff in atoms:
        contribution = coeff * local_j[shift_index] * (k ** total_k_power) * (mu ** total_mu_power)
        total += contribution
        abs_contribution = abs(contribution)
        contribution_abs_sum += abs_contribution
        largest_abs_contribution = max(largest_abs_contribution, abs_contribution)

    if not return_stats:
        return total

    condition_proxy = float("inf") if abs(total) == 0.0 and contribution_abs_sum > 0.0 else contribution_abs_sum / max(abs(total), 1e-300)
    return total, {
        **jcache.stats(),
        "unique_grouped_atoms": prepared["unique_grouped_atoms"],
        "largest_abs_contribution": largest_abs_contribution,
        "condition_proxy": condition_proxy,
    }


def evaluate_descriptor_dict_runtime_flattened(descriptor: dict, nu1, nu2, mu, k, *, return_stats: bool = False):
    grouped = flatten_descriptor(descriptor)
    return evaluate_flattened_descriptor(grouped, nu1=nu1, nu2=nu2, mu=mu, k=k, return_stats=return_stats)


def evaluate_descriptor_dict_runtime_flattened_prepared(descriptor: dict, nu1, nu2, mu, k, *, return_stats: bool = False):
    prepared = prepare_flattened_descriptor(descriptor)
    return evaluate_preflattened_descriptor(prepared, nu1=nu1, nu2=nu2, mu=mu, k=k, return_stats=return_stats)
