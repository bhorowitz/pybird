"""FFTLog-style descriptor-to-table writer for LyA channels.

This bridges descriptor JSON files and packed `LYA_M13__*.dat` numerical tables
following the storage convention in `lya_matrix_generation_guide.txt`:

- fixed FFTLog grid,
- Hermitian fill,
- lower-triangular packing,
- real block followed by imag block on disk.

The evaluator is still a development implementation of the LOS master library,
but the output format now matches the intended backend-facing contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lya_descriptor_runtime import (
    descriptor_k_powers,
    evaluate_descriptor_kernel,
    evaluate_descriptor_physical,
    evaluate_descriptor_dict_runtime,
    evaluate_preflattened_descriptor,
    prepare_flattened_descriptor,
    split_descriptor_by_k_power,
)
from lya_master_library import COMPONENT_SHIFTS, J, RADIAL_1D_KERNELS


def load_descriptor(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


DEFAULT_NMAX = 32
DEFAULT_FFTLOGBIAS = -1.6
DEFAULT_K0 = 5.0e-5
DEFAULT_KMAX = 1.0e2
DEFAULT_RADIAL_SAMPLES = 256


def descriptor_supports_prepared_runtime(descriptor: dict) -> bool:
    return all(term["master_component"] in COMPONENT_SHIFTS for term in descriptor["terms"])


def build_descriptor_evaluator(descriptor: dict, backend: str = "auto"):
    if backend not in {"auto", "old", "prepared"}:
        raise ValueError(f"Unsupported backend selector {backend!r}")

    if backend == "prepared":
        if not descriptor_supports_prepared_runtime(descriptor):
            raise ValueError(
                f"Descriptor {descriptor.get('channel_name', '<unknown>')} is not fully covered by COMPONENT_SHIFTS"
            )
        prepared = prepare_flattened_descriptor(descriptor)

        def evaluator(*, nu1, nu2, mu, k):
            return evaluate_preflattened_descriptor(prepared, nu1=nu1, nu2=nu2, mu=mu, k=k)

        return "prepared-flattened", evaluator

    if backend == "auto" and descriptor_supports_prepared_runtime(descriptor):
        prepared = prepare_flattened_descriptor(descriptor)

        def evaluator(*, nu1, nu2, mu, k):
            return evaluate_preflattened_descriptor(prepared, nu1=nu1, nu2=nu2, mu=mu, k=k)

        return "prepared-flattened", evaluator

    def evaluator(*, nu1, nu2, mu, k):
        return evaluate_descriptor_dict_runtime(descriptor, nu1=nu1, nu2=nu2, mu=mu, k=k)

    return "runtime-fallback", evaluator


def build_physical_descriptor_evaluator(descriptor: dict, backend: str = "auto"):
    if backend not in {"auto", "old", "prepared"}:
        raise ValueError(f"Unsupported backend selector {backend!r}")

    if backend in {"auto", "prepared"} and descriptor_supports_prepared_runtime(descriptor):
        prepared = prepare_flattened_descriptor(descriptor)

        def evaluator(*, nu1, nu2, mu, k):
            return evaluate_preflattened_descriptor(prepared, nu1=nu1, nu2=nu2, mu=mu, k=k)

        return "prepared-flattened", evaluator

    def evaluator(*, nu1, nu2, mu, k):
        return evaluate_descriptor_physical(descriptor, nu1=nu1, nu2=nu2, mu=mu, k=k)

    return "runtime-fallback", evaluator


def fftlog_grid(nmax: int = DEFAULT_NMAX, fftlogbias: float = DEFAULT_FFTLOGBIAS, k0: float = DEFAULT_K0, kmax: float = DEFAULT_KMAX) -> np.ndarray:
    delta = np.log(kmax / k0) / (nmax - 1)
    js = np.arange(-nmax / 2, nmax / 2 + 1, 1)
    return fftlogbias + 2j * np.pi * js / nmax / delta


def descriptor_matrix(
    descriptor: dict,
    nmax: int = DEFAULT_NMAX,
    fftlogbias: float = DEFAULT_FFTLOGBIAS,
    k0: float = DEFAULT_K0,
    kmax: float = DEFAULT_KMAX,
    mu: float = 0.5,
    k: float = 0.2,
    backend: str = "auto",
) -> np.ndarray:
    etam = fftlog_grid(nmax=nmax, fftlogbias=fftlogbias, k0=k0, kmax=kmax)
    size = nmax + 1
    mat = np.zeros((size, size), dtype=np.complex128)
    half = nmax / 2
    _, evaluator = build_physical_descriptor_evaluator(descriptor, backend=backend)

    for j1 in range(size):
        for j2 in range(size):
            if j1 - half < 1:
                nu1 = -0.5 * etam[j1]
                nu2 = -0.5 * etam[j2]
                mat[j1, j2] = complex(evaluator(nu1=nu1, nu2=nu2, mu=mu, k=k))
            else:
                mat[j1, j2] = np.conjugate(mat[nmax - j1, nmax - j2])

    return mat


def descriptor_matrix_kernel(
    descriptor: dict,
    nmax: int = DEFAULT_NMAX,
    fftlogbias: float = DEFAULT_FFTLOGBIAS,
    k0: float = DEFAULT_K0,
    kmax: float = DEFAULT_KMAX,
) -> np.ndarray:
    etam = fftlog_grid(nmax=nmax, fftlogbias=fftlogbias, k0=k0, kmax=kmax)
    size = nmax + 1
    mat = np.zeros((size, size), dtype=np.complex128)
    half = nmax / 2

    if len(descriptor_k_powers(descriptor)) > 1:
        raise ValueError("descriptor_matrix_kernel requires a single fixed-k-power descriptor")

    for j1 in range(size):
        for j2 in range(size):
            if j1 - half < 1:
                nu1 = -0.5 * etam[j1]
                nu2 = -0.5 * etam[j2]
                mat[j1, j2] = complex(evaluate_descriptor_kernel(descriptor, nu1=nu1, nu2=nu2))
            else:
                mat[j1, j2] = np.conjugate(mat[nmax - j1, nmax - j2])

    return mat


def pack_lower_triangular(mat: np.ndarray) -> np.ndarray:
    size = mat.shape[0]
    packed = np.zeros((size * (size + 1)) // 2, dtype=np.complex128)
    for i in range(size):
        for j in range(i + 1):
            packed[i + (2 * size - 1 - j) * j // 2] = mat[i, j]
    return packed


def write_packed_fftlog_matrices(
    descriptor_path: Path,
    output_dir: Path,
    *,
    nmax: int = DEFAULT_NMAX,
    fftlogbias: float = DEFAULT_FFTLOGBIAS,
    k0: float = DEFAULT_K0,
    kmax: float = DEFAULT_KMAX,
    backend: str = "auto",
) -> list[tuple[Path, Path]]:
    descriptor = load_descriptor(descriptor_path)
    split = split_descriptor_by_k_power(descriptor)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, Path]] = []

    for k_power, sector_descriptor in sorted(split.items()):
        mat = descriptor_matrix_kernel(sector_descriptor, nmax=nmax, fftlogbias=fftlogbias, k0=k0, kmax=kmax)
        packed = pack_lower_triangular(mat)
        output_path = output_dir / (
            f"LYA_M13__{descriptor['channel_name']}__MU{descriptor['mu_power']}__KPOW{k_power}.dat"
        )
        stacked = np.concatenate([packed.real, packed.imag])
        np.savetxt(output_path, stacked)

        metadata = {
            "format": "fftlog-packed-lower-triangular",
            "artifact_role": "fftlog_packed_matrix",
            "generation_mode": "production_kernel",
            "sector": descriptor.get("sector"),
            "channel_name": descriptor.get("channel_name"),
            "channel_id": descriptor.get("channel_id"),
            "mu_power": descriptor.get("mu_power"),
            "k_power": int(k_power),
            "storage": {
                "packing": "lower-triangular",
                "layout": "real-block-then-imag-block",
                "nmax": nmax,
                "matrix_size": nmax + 1,
                "packed_size": int(packed.size),
                "fftlog_bias": fftlogbias,
                "k0": k0,
                "kmax": kmax,
                "evaluation_backend": "kernel-only",
            },
            "source_descriptor": str(descriptor_path),
        }
        metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        results.append((output_path, metadata_path))

    return results


def radial_k_grid(sample_count: int = DEFAULT_RADIAL_SAMPLES, k0: float = DEFAULT_K0, kmax: float = DEFAULT_KMAX) -> np.ndarray:
    return np.geomspace(k0, kmax, sample_count)


def descriptor_radial_vector(
    descriptor: dict,
    sample_count: int = DEFAULT_RADIAL_SAMPLES,
    *,
    nu1: complex = 0.8 + 0.1j,
    nu2: complex = 0.7 - 0.05j,
    mu: float = 1.0,
    k0: float = DEFAULT_K0,
    kmax: float = DEFAULT_KMAX,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    k_grid = radial_k_grid(sample_count=sample_count, k0=k0, kmax=kmax)
    _, evaluator = build_descriptor_evaluator(descriptor, backend=backend)
    values = np.asarray(
    [complex(evaluator(nu1=nu1, nu2=nu2, mu=mu, k=float(k))) for k in k_grid],
        dtype=np.complex128,
    )
    return k_grid, values


def write_debug_radial_probe(
    descriptor_path: Path,
    output_path: Path,
    *,
    sample_count: int = DEFAULT_RADIAL_SAMPLES,
    nu1: complex = 0.8 + 0.1j,
    nu2: complex = 0.7 - 0.05j,
    mu: float = 1.0,
    k0: float = DEFAULT_K0,
    kmax: float = DEFAULT_KMAX,
    backend: str = "auto",
) -> tuple[Path, Path]:
    descriptor = load_descriptor(descriptor_path)
    evaluator_name, _ = build_descriptor_evaluator(descriptor, backend=backend)
    k_grid, values = descriptor_radial_vector(
        descriptor,
        sample_count=sample_count,
        nu1=nu1,
        nu2=nu2,
        mu=mu,
        k0=k0,
        kmax=kmax,
        backend=backend,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.column_stack([k_grid, values.real, values.imag])
    np.savetxt(output_path, stacked)

    metadata = {
    "format": "lya-radial-operator-table-v1",
    "artifact_role": "debug_radial_probe",
    "generation_mode": "debug_physical",
        "sector": descriptor.get("sector"),
        "channel_name": descriptor.get("channel_name"),
        "channel_id": descriptor.get("channel_id"),
        "mu_power": descriptor.get("mu_power"),
        "storage": {
            "layout": "columns:k,re,im",
            "sample_count": int(sample_count),
            "k0": float(k0),
            "kmax": float(kmax),
            "evaluation_mu": float(mu),
            "evaluation_nu1_re": float(np.real(nu1)),
            "evaluation_nu1_im": float(np.imag(nu1)),
            "evaluation_nu2_re": float(np.real(nu2)),
            "evaluation_nu2_im": float(np.imag(nu2)),
            "evaluation_backend": evaluator_name,
        },
        "source_descriptor": str(descriptor_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path, metadata_path


def sampled_descriptor_vector(descriptor: dict, sample_count: int = 16, mu: float = 0.5, k: float = 0.2) -> np.ndarray:
    """Backward-compatible helper kept for older smoke tests/debug scripts."""
    nu1_vals = np.linspace(0.65, 0.95, sample_count)
    nu2_vals = np.linspace(0.55, 0.85, sample_count)
    _, evaluator = build_descriptor_evaluator(descriptor, backend="auto")
    values = []
    for nu1, nu2 in zip(nu1_vals, nu2_vals):
        val = evaluator(nu1=nu1, nu2=nu2, mu=mu, k=k)
        values.append(complex(val))
    return np.asarray(values, dtype=np.complex128)


def write_sampled_vector(descriptor_path: Path, output_path: Path, sample_count: int = 16, mu: float = 0.5, k: float = 0.2) -> Path:
    descriptor = load_descriptor(descriptor_path)
    vec = sampled_descriptor_vector(descriptor, sample_count=sample_count, mu=mu, k=k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.concatenate([vec.real, vec.imag])
    np.savetxt(output_path, stacked)
    return output_path


def write_packed_fftlog_matrix(*args, **kwargs):
    raise RuntimeError(
        "write_packed_fftlog_matrix() has been replaced by write_packed_fftlog_matrices(); production packed files must now be generated per fixed k-power sector"
    )


write_backend_radial_table = write_debug_radial_probe


def _eval_master_for_p13(component: str, nu1: complex, dn2_int: int) -> complex:
    """Evaluate a master component at fixed integer nu2, handling Gamma poles.

    In the T13 PROJ context, nu2 is a fixed integer (delta_nu2 from the
    descriptor), not a free FFTLog variable.  J(nu1, nu2) = 0 whenever nu2
    is a non-positive integer (1/Gamma(nu2) = 0 in the J definition).

    Parameters
    ----------
    component : str
        Name in COMPONENT_SHIFTS (e.g. "A2", "A3", "B4", ...).
    nu1 : complex
        Already-shifted Mellin variable (nu1_j + delta_nu1 from descriptor).
    dn2_int : int
        Fixed integer nu2 shift (delta_nu2 from descriptor).
    """
    shifts = COMPONENT_SHIFTS[component]
    result = 0 + 0j
    for weight, s1, s2 in shifts:
        nu2_eff = dn2_int + int(s2)
        if nu2_eff <= 0:   # non-positive integer -> 1/Gamma(nu2_eff) = 0 -> J = 0
            continue
        result += weight * J(nu1 + s1, nu2_eff)
    return result


def write_shifted_j_p13_vectors(
    descriptor_path: Path,
    output_dir: Path,
    *,
    nmax: int = DEFAULT_NMAX,
    fftlogbias: float = DEFAULT_FFTLOGBIAS,
    k0: float = DEFAULT_K0,
    kmax: float = DEFAULT_KMAX,
) -> list[tuple[Path, Path]]:
    """Write 1D P13 radial vectors per k-power sector for PROJ-type T13 channels.

    Unlike the 2D packed-matrix format, PROJ channels have a fixed (integer)
    nu2 for the second loop variable after the angular integration.  The full
    P13 integral reduces to a standard 1D sum:

        P13(k) = 2 * P_lin(k) * k^{3+k_power}
                 * Re[ sum_n c_n * k^{Pow_n} * V_{k_power}[n] ]

    where V_{k_power}[n] = sum_terms coeff * master(nu_n + delta_nu1, delta_nu2).

    Writes one file per k-power sector with filename:
        LYA_M13__{channel}__MU{mu}__KPOW{kp}__P13VEC.dat

    Format: real block (nmax+1 values) then imag block (nmax+1 values).
    """
    descriptor = load_descriptor(descriptor_path)
    split = split_descriptor_by_k_power(descriptor)
    output_dir.mkdir(parents=True, exist_ok=True)
    etam = fftlog_grid(nmax=nmax, fftlogbias=fftlogbias, k0=k0, kmax=kmax)
    size = nmax + 1
    results: list[tuple[Path, Path]] = []

    channel_name = descriptor["channel_name"]
    mu_power = descriptor["mu_power"]

    for k_power, sector in sorted(split.items()):
        vec = np.zeros(size, dtype=np.complex128)
        for j in range(size):
            nu1_j = -0.5 * etam[j]
            for term in sector["terms"]:
                coeff = term["coeff_rational"]["num"] / term["coeff_rational"]["den"]
                dn1 = int(term["delta_nu1"])
                dn2 = int(term["delta_nu2"])
                mc = term["master_component"]
                vec[j] += coeff * _eval_master_for_p13(mc, nu1_j + dn1, dn2)

        output_path = output_dir / (
            f"LYA_M13__{channel_name}__MU{mu_power}__KPOW{k_power}__P13VEC.dat"
        )
        stacked = np.concatenate([vec.real, vec.imag])
        np.savetxt(output_path, stacked)

        metadata = {
            "format": "shifted-j-p13-1d",
            "artifact_role": "p13_1d_vector",
            "generation_mode": "shifted_j_p13_vector",
            "sector": descriptor.get("sector"),
            "channel_name": channel_name,
            "channel_id": descriptor.get("channel_id"),
            "mu_power": mu_power,
            "k_power": int(k_power),
            "storage": {
                "layout": "real-block-then-imag-block",
                "nmax": nmax,
                "vector_size": size,
                "fftlog_bias": fftlogbias,
                "k0": k0,
                "kmax": kmax,
            },
            "source_descriptor": str(descriptor_path),
        }
        metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        results.append((output_path, metadata_path))

    return results


def write_radial_1d_table(
    descriptor_path: Path,
    output_dir: Path,
    *,
    nmax: int = DEFAULT_NMAX,
    fftlogbias: float = DEFAULT_FFTLOGBIAS,
    k0: float = DEFAULT_K0,
    kmax: float = DEFAULT_KMAX,
) -> tuple[Path, Path]:
    """Write a 1D radial kernel vector for F3/G3 type T13 channels.

    Unlike the 2D packed-matrix format used for shifted-J channels, the
    F3 and G3 SPT kernels are isotropic and their P13 contribution reduces
    to a 1D Mellin-space radial vector (one complex value per FFTLog mode).

    Output file format: real block then imag block, length (nmax+1) each.

    Returns
    -------
    (data_path, metadata_path)
    """
    descriptor = load_descriptor(descriptor_path)
    radial_kernel_type = descriptor.get("radial_kernel_type")
    if radial_kernel_type not in RADIAL_1D_KERNELS:
        raise ValueError(
            f"Descriptor {descriptor['channel_name']!r} has unsupported "
            f"radial_kernel_type={radial_kernel_type!r}; "
            f"expected one of {list(RADIAL_1D_KERNELS)}"
        )
    kernel_fn = RADIAL_1D_KERNELS[radial_kernel_type]

    etam = fftlog_grid(nmax=nmax, fftlogbias=fftlogbias, k0=k0, kmax=kmax)
    size = nmax + 1
    vec = np.zeros(size, dtype=np.complex128)
    for j in range(size):
        nu1 = -0.5 * etam[j]
        vec[j] = complex(kernel_fn(nu1))

    output_dir.mkdir(parents=True, exist_ok=True)
    channel_name = descriptor["channel_name"]
    mu_power = descriptor["mu_power"]
    output_path = output_dir / f"LYA_M13__{channel_name}__MU{mu_power}__RADIAL1D.dat"
    stacked = np.concatenate([vec.real, vec.imag])
    np.savetxt(output_path, stacked)

    metadata = {
        "format": "fftlog-radial-1d",
        "artifact_role": "radial_1d_vector",
        "generation_mode": "radial_1d_kernel",
        "sector": descriptor.get("sector"),
        "channel_name": channel_name,
        "channel_id": descriptor.get("channel_id"),
        "mu_power": mu_power,
        "radial_kernel_type": radial_kernel_type,
        "storage": {
            "layout": "real-block-then-imag-block",
            "nmax": nmax,
            "vector_size": size,
            "fftlog_bias": fftlogbias,
            "k0": k0,
            "kmax": kmax,
        },
        "source_descriptor": str(descriptor_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path, metadata_path
