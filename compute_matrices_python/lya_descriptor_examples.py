"""Prototype non-empty LyA descriptors.

These are deliberately small hand-authored examples for exercising the new
descriptor pipeline before the full symbolic reducer exists. They are not yet
the final physics reduction for the corresponding LyA operators.
"""

from __future__ import annotations

from lya_channels import T13_CHANNEL_MAP
from lya_descriptor_schema import ChannelDescriptor, DescriptorTerm


def prototype_t13_deltaeta_mu2() -> ChannelDescriptor:
    channel = T13_CHANNEL_MAP["T13_B_DELTAETA"]
    return ChannelDescriptor(
        version="1.0",
        sector="13",
        channel_name=channel.name,
        channel_id=channel.channel_id,
        mu_power=2,
        k_power_prefactor=1,
        radial_rank=0,
        terms=(
            DescriptorTerm(
                coeff_rational={"num": -1, "den": 2},
                coeff_float=None,
                los_rank_r=0,
                delta_nu1=0,
                delta_nu2=0,
                master_component="J",
                extra_k_power=0,
                extra_mu_power=0,
                numerator_tag="prototype_deltaeta_base",
                notes="Prototype seed term for descriptor/evaluator plumbing only",
            ),
            DescriptorTerm(
                coeff_rational={"num": 1, "den": 6},
                coeff_float=None,
                los_rank_r=2,
                delta_nu1=1,
                delta_nu2=0,
                master_component="A6",
                extra_k_power=0,
                extra_mu_power=0,
                numerator_tag="prototype_deltaeta_los_rank2",
                notes="Prototype higher-r LOS term for evaluator testing",
            ),
        ),
        metadata={
            "status": "prototype-nonempty",
            "physics_status": "not-final",
            "notes": "Hand-authored prototype used to validate descriptor and evaluator plumbing before symbolic reduction lands.",
        },
    )
