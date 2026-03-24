"""Generate first-pass LyA matrix descriptors.

This is the first concrete artifact from `lya_matrix_generation_guide.txt`.
For now it emits validated placeholder descriptors for the recommended initial
13-channel subset, giving the symbolic reduction layer and backend a stable
contract to build around.
"""

from __future__ import annotations

from pathlib import Path
import json

from lya_channels import INITIAL_T13_SUBSET, T13_CHANNEL_MAP
from lya_descriptor_schema import ChannelDescriptor, write_descriptor
from lya_reduction import (
    reduce_t13_deltaeta_mu2,
    reduce_t13_proj_b1_mu0,
    reduce_t13_proj_b1_mu2,
    reduce_t13_proj_beta_mu2,
    reduce_t13_proj_beta_mu4,
)
from lya_sympy_bridge import generate_sympy_13_descriptor_dicts


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "lya_descriptors" / "13"


REDUCER_OVERRIDES = {
    ("T13_B_DELTAETA", 2): reduce_t13_deltaeta_mu2,
    ("T13_B_PROJ_B1", 0): reduce_t13_proj_b1_mu0,
    ("T13_B_PROJ_B1", 2): reduce_t13_proj_b1_mu2,
    ("T13_B_PROJ_BETA", 2): reduce_t13_proj_beta_mu2,
    ("T13_B_PROJ_BETA", 4): reduce_t13_proj_beta_mu4,
}

BACKEND_IMPORTED_CHANNELS = {
    "T13_B_F3_B1": "standard CLASS-PT P13 F3 radial kernel path",
    "T13_B_G3_BETA": "standard CLASS-PT P13 G3 radial kernel path",
}

PHYSICALLY_ZERO_CHANNEL_MU = {
    ("T13_B_KKPAR", 4): "Binomial conservation in zkq = k*mu - zq forbids the explicit-mu / LOS-rank combinations needed to reach total mu^4 for KKPAR; the exact reduced coefficient is zero.",
    ("T13_B_PI2PAR_MAIN", 4): "Binomial conservation in zkq = k*mu - zq, together with qdotkmq carrying no LOS structure, forbids the combinations needed to reach total mu^4 for PI2PAR_MAIN; the exact reduced coefficient is zero.",
    ("T13_B_DELTAPI2PAR", 4): "The kernel is proportional to zkq^2 with no additional LOS source, so the exact reduction emits only mu^0 and mu^2; the mu^4 slot is identically zero.",
    ("T13_B_ETAPI2PAR", 6): "The kernel is proportional to zq^2 * zkq^2, so the exact reduction tops out at mu^4; the mu^6 slot is identically zero.",
    ("T13_B_G2_COMPOSITE", 2): "The kernel qdotkmq^2 - q^2*kmq^2 is isotropic in the LOS sense and the exact reduction emits only mu^0; the mu^2 slot is identically zero.",
    ("T13_B_G2_COMPOSITE", 4): "The kernel qdotkmq^2 - q^2*kmq^2 is isotropic in the LOS sense and the exact reduction emits only mu^0; the mu^4 slot is identically zero.",
    ("T13_B_KPI2PAR_AUX", 4): "The auxiliary KPi2_parallel kernel is proportional to zkq^2 with no extra LOS factor, so the exact reduction emits only mu^0 and mu^2; the mu^4 slot is identically zero.",
    ("T13_B_KPI2PAR_AUX", 6): "The auxiliary KPi2_parallel kernel is proportional to zkq^2 with no extra LOS factor, so the exact reduction emits only mu^0 and mu^2; the mu^6 slot is identically zero.",
    ("T13_B_KPI2PAR_MAIN", 4): "The kernel qdotkmq * zq * zkq carries only two LOS factors beyond isotropic pieces, so the exact reduction emits mu^0 and mu^2 only; the mu^4 slot is identically zero.",
    ("T13_B_KPI2PAR_MAIN", 6): "The kernel qdotkmq * zq * zkq carries only two LOS factors beyond isotropic pieces, so the exact reduction emits mu^0 and mu^2 only; the mu^6 slot is identically zero.",
    ("T13_B_PI2PAR_CUBIC", 6): "The kernel zq * zkq^3 contains four LOS factors total, so the exact reduction tops out at mu^4; the mu^6 slot is identically zero.",
    ("T13_B_PI3PAR", 4): "The exact PI3_parallel reduction combines qdotkmq*zq*zkq and an explicit mu^2 isotropic term, yielding only mu^0 and mu^2 after angular reduction; the mu^4 slot is identically zero.",
    ("T13_B_PI3PAR", 6): "The exact PI3_parallel reduction combines qdotkmq*zq*zkq and an explicit mu^2 isotropic term, yielding only mu^0 and mu^2 after angular reduction; the mu^6 slot is identically zero.",
}


def build_placeholder_t13_descriptor(channel_name: str, mu_power: int) -> ChannelDescriptor:
    channel = T13_CHANNEL_MAP[channel_name]
    if channel_name in BACKEND_IMPORTED_CHANNELS:
        return ChannelDescriptor(
            version="1.0",
            sector="13",
            channel_name=channel.name,
            channel_id=channel.channel_id,
            mu_power=mu_power,
            k_power_prefactor=0,
            radial_rank=0,
            terms=(),
            metadata={
                "status": "backend-import",
                "import_source": BACKEND_IMPORTED_CHANNELS[channel_name],
                "notes": "Intentional interface boundary: this channel is provided by the existing CLASS-PT F3/G3 infrastructure rather than the LOS shifted-J descriptor generator.",
            },
        )
    if (channel_name, mu_power) in PHYSICALLY_ZERO_CHANNEL_MU:
        return ChannelDescriptor(
            version="1.0",
            sector="13",
            channel_name=channel.name,
            channel_id=channel.channel_id,
            mu_power=mu_power,
            k_power_prefactor=0,
            radial_rank=0,
            terms=(),
            metadata={
                "status": "physically-zero",
                "notes": PHYSICALLY_ZERO_CHANNEL_MU[(channel_name, mu_power)],
            },
        )
    return ChannelDescriptor(
        version="1.0",
        sector="13",
        channel_name=channel.name,
        channel_id=channel.channel_id,
        mu_power=mu_power,
        k_power_prefactor=0,
        radial_rank=0,
        terms=(),
        metadata={
            "status": "placeholder",
            "notes": channel.notes,
            "next_step": "fill terms from symbolic reducer described in lya_matrix_generation_guide.txt",
        },
    )


def main() -> None:
    written = []
    for channel_name in INITIAL_T13_SUBSET:
        channel = T13_CHANNEL_MAP[channel_name]
        sympy_descriptors = generate_sympy_13_descriptor_dicts(channel_name)
        if sympy_descriptors:
            emitted_mu_powers = set()
            for payload in sympy_descriptors:
                mu_power = payload["mu_power"]
                emitted_mu_powers.add(mu_power)
                output_path = OUT_DIR / f"LYA_T13__{channel.name}__MU{mu_power}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                written.append(output_path)
            for mu_power in channel.mu_powers:
                if mu_power in emitted_mu_powers:
                    continue
                descriptor = build_placeholder_t13_descriptor(channel_name, mu_power)
                output_path = OUT_DIR / f"LYA_T13__{channel.name}__MU{mu_power}.json"
                write_descriptor(descriptor, output_path)
                written.append(output_path)
            continue
        for mu_power in channel.mu_powers:
            reducer = REDUCER_OVERRIDES.get((channel_name, mu_power))
            if reducer is not None:
                descriptor = reducer()
            else:
                descriptor = build_placeholder_t13_descriptor(channel_name, mu_power)
            output_path = OUT_DIR / f"LYA_T13__{channel.name}__MU{mu_power}.json"
            write_descriptor(descriptor, output_path)
            written.append(output_path)
    print(f"Wrote {len(written)} LyA descriptor files to {OUT_DIR}")


if __name__ == "__main__":
    main()