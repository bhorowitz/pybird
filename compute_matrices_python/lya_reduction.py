"""Prototype symbolic-reduction layer for LyA descriptors.

This is the first reducer-backed step from `lya_matrix_generation_guide.txt`.
It does not attempt the full Eq. (3.20) expansion yet. Instead, it provides a
small, structured reduction output for a few prototype channels so the
descriptor pipeline becomes reproducible rather than purely hand-authored.
"""

from __future__ import annotations

from lya_channels import T13_CHANNEL_MAP
from lya_descriptor_schema import ChannelDescriptor, DescriptorTerm


def _build_t13_descriptor(channel_name: str, mu_power: int, k_power_prefactor: int, reduced_terms, reduction_source: str, notes: str) -> ChannelDescriptor:
    channel = T13_CHANNEL_MAP[channel_name]
    return ChannelDescriptor(
        version="1.0",
        sector="13",
        channel_name=channel.name,
        channel_id=channel.channel_id,
        mu_power=mu_power,
        k_power_prefactor=k_power_prefactor,
        radial_rank=0,
        terms=tuple(DescriptorTerm(**term) for term in reduced_terms),
        metadata={
            "status": "reducer-prototype",
            "physics_status": "not-final",
            "reduction_source": reduction_source,
            "notes": notes,
        },
    )


def reduce_t13_deltaeta_mu2() -> ChannelDescriptor:
    """Return a prototype reduced descriptor for `T13_B_DELTAETA` at `mu^2`.

    Physics status:
    - reducer-backed prototype
    - not the final analytic reduction of Eq. (3.20)
    - intended to exercise the descriptor contract with named reduced terms
    """

    reduced_terms = (
        {
            "coeff_rational": {"num": -1, "den": 2},
            "coeff_float": None,
            "los_rank_r": 0,
            "delta_nu1": 0,
            "delta_nu2": 0,
            "master_component": "J",
            "extra_k_power": 0,
            "extra_mu_power": 0,
            "numerator_tag": "reduced_deltaeta_seed_base",
            "notes": "Reducer-backed prototype base term",
        },
        {
            "coeff_rational": {"num": 1, "den": 6},
            "coeff_float": None,
            "los_rank_r": 2,
            "delta_nu1": 1,
            "delta_nu2": 0,
            "master_component": "A6",
            "extra_k_power": 0,
            "extra_mu_power": 0,
            "numerator_tag": "reduced_deltaeta_seed_los_rank2",
            "notes": "Reducer-backed prototype higher-r LOS term",
        },
    )

    return _build_t13_descriptor(
        channel_name="T13_B_DELTAETA",
        mu_power=2,
        k_power_prefactor=1,
        reduced_terms=reduced_terms,
        reduction_source="prototype_reduce_t13_deltaeta_mu2",
        notes="Minimal reducer-backed descriptor used to validate the transition from static examples to a reduction layer.",
    )


def reduce_t13_proj_b1_mu0() -> ChannelDescriptor:
    reduced_terms = (
        {
            "coeff_rational": {"num": 1, "den": 1},
            "coeff_float": None,
            "los_rank_r": 0,
            "delta_nu1": 0,
            "delta_nu2": 0,
            "master_component": "J",
            "extra_k_power": 0,
            "extra_mu_power": 0,
            "numerator_tag": "reduced_proj_b1_mu0_base",
            "notes": "Prototype projection base term for mu^0",
        },
    )
    return _build_t13_descriptor(
        channel_name="T13_B_PROJ_B1",
        mu_power=0,
        k_power_prefactor=0,
        reduced_terms=reduced_terms,
        reduction_source="prototype_reduce_t13_proj_b1_mu0",
        notes="Reducer-backed prototype for projection b1 mu^0 channel.",
    )


def reduce_t13_proj_b1_mu2() -> ChannelDescriptor:
    reduced_terms = (
        {
            "coeff_rational": {"num": -1, "den": 3},
            "coeff_float": None,
            "los_rank_r": 2,
            "delta_nu1": 0,
            "delta_nu2": 0,
            "master_component": "A6",
            "extra_k_power": 0,
            "extra_mu_power": 0,
            "numerator_tag": "reduced_proj_b1_mu2_rank2",
            "notes": "Prototype projection higher-r mu^2 term for b1",
        },
    )
    return _build_t13_descriptor(
        channel_name="T13_B_PROJ_B1",
        mu_power=2,
        k_power_prefactor=1,
        reduced_terms=reduced_terms,
        reduction_source="prototype_reduce_t13_proj_b1_mu2",
        notes="Reducer-backed prototype for projection b1 mu^2 channel.",
    )


def reduce_t13_proj_beta_mu2() -> ChannelDescriptor:
    reduced_terms = (
        {
            "coeff_rational": {"num": -2, "den": 3},
            "coeff_float": None,
            "los_rank_r": 2,
            "delta_nu1": 0,
            "delta_nu2": 1,
            "master_component": "A6",
            "extra_k_power": 0,
            "extra_mu_power": 0,
            "numerator_tag": "reduced_proj_beta_mu2_rank2",
            "notes": "Prototype projection mu^2 term for beta",
        },
    )
    return _build_t13_descriptor(
        channel_name="T13_B_PROJ_BETA",
        mu_power=2,
        k_power_prefactor=1,
        reduced_terms=reduced_terms,
        reduction_source="prototype_reduce_t13_proj_beta_mu2",
        notes="Reducer-backed prototype for projection beta mu^2 channel.",
    )


def reduce_t13_proj_beta_mu4() -> ChannelDescriptor:
    reduced_terms = (
        {
            "coeff_rational": {"num": 1, "den": 5},
            "coeff_float": None,
            "los_rank_r": 2,
            "delta_nu1": 1,
            "delta_nu2": 0,
            "master_component": "A6",
            "extra_k_power": 0,
            "extra_mu_power": 0,
            "numerator_tag": "reduced_proj_beta_mu4_rank2",
            "notes": "Prototype projection mu^4 term for beta",
        },
    )
    return _build_t13_descriptor(
        channel_name="T13_B_PROJ_BETA",
        mu_power=4,
        k_power_prefactor=1,
        reduced_terms=reduced_terms,
        reduction_source="prototype_reduce_t13_proj_beta_mu4",
        notes="Reducer-backed prototype for projection beta mu^4 channel.",
    )
