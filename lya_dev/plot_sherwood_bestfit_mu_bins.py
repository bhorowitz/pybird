#!/usr/bin/env python3
"""Plot Sherwood z=2.8 best-fit LyA P3D components in representative mu bins.

This uses PyBird/JAX for the tree-level LyA result and writes
`lya-dev/output/sherwood_bestfit_k2pk_mu_bins.png` plus a small `.npz`
cache of the sampled arrays.

The parameter point is the paper-style z=2.8 Sherwood best fit already
used elsewhere in this workspace for direct comparison work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
try:
    from scipy.stats import binned_statistic as _binned_statistic
except Exception:
    _binned_statistic = None

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT
OUTPUT_DIR = ROOT / "lya-dev" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#"b1": -0.2168,
#    "b_eta": 0.324,

SHERWOOD_Z28_PARAMS = {
    "z": 2.8,
    "b1": -0.24168,
    "b_eta": 0.324,
}

# The simulation box size and grid (assumed 160 Mpc/h, 2048^3)
BOX_SIZE = 160.0 # Mpc/h

MU_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.01], dtype=float)
MU_BIN_CENTERS = 0.5 * (MU_EDGES[:-1] + MU_EDGES[1:])   # [0.125, 0.375, 0.625, 0.88]
MU_BIN_LABELS = [
    r"$\mu \in [0.00,\,0.25)$",
    r"$\mu \in [0.25,\,0.50)$",
    r"$\mu \in [0.50,\,0.75)$",
    r"$\mu \in [0.75,\,1.00)$",
]
K_PLOT_BINS = np.logspace(np.log10(0.04), np.log10(3.0), 14)
K_GRID = np.geomspace(1.0e-2, 5.0, 240)


BASELINE_COSMO = {
    "h": 0.678,
    "omega_b": 0.0482 * 0.678**2,
    "omega_cdm": (0.308 - 0.0482) * 0.678**2,
    "ln10^{10}A_s": 3.044,
    "n_s": 0.961,
}


def _build_pybird_tree_spectrum(k_grid: np.ndarray, z: float, b1: float, b_eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (k, P_tree(mu, k)) using the paper1 LyA tree-level convention.

    The tree-level flux power is
    P_tree(k, mu) = (b1 - b_eta * f(z) * mu^2)^2 P_lin(k),
    with k in h/Mpc and P in (Mpc/h)^3.
    """
    class_cache = OUTPUT_DIR / "class_plin_z28.npz"
    if class_cache.exists():
        print(f"Loading CLASS P_lin from {class_cache}")
        data_cache = np.load(class_cache)
        kk_class = data_cache["kk"]
        pk_class = data_cache["pk"]
        f_class = float(data_cache["f"])
        
        # Interpolate CLASS P_lin to k_grid
        pk_lin = np.interp(k_grid, kk_class, pk_class, left=0, right=0)
        f = f_class
    else:
        print("CLASS cache not found, falling back to PyBird Symbolic")
        from pybird.cosmo import Cosmo
        from pybird.correlator import Correlator

        corr = Correlator()
        corr.set({
            "output": "bPk",
            "z": z,
            "with_bias": True,
            "with_resum": False,
            "with_time": False,
            "with_exact_time": False,
            "with_redshift_bin": False,
            "with_ap": False,
            "with_binning": False,
            "with_survey_mask": False,
            "with_fibercol": False,
            "with_wedge": False,
            "kmin": float(np.min(k_grid)),
            "kmax": 10.0,
            "multipole": 3,
            "eft_basis": "eftoflss",
            "km": 1.0,
            "kr": 1.0,
        })

        cosmo = Cosmo(corr.c)
        cosmo_dict = cosmo.set_cosmo(BASELINE_COSMO, module="Symbolic")
        cosmo_dict["bias"] = {"b1": b1, "b_eta": b_eta, "b2": 0.0, "b3": 0.0, "b4": 0.0, "cct": 0.0, "cr1": 0.0, "cr2": 0.0}
        
        corr.compute(cosmo_dict=cosmo_dict, do_core=True, do_survey_specific=False)
        f = float(corr.bird.f)
        pk_lin = np.interp(k_grid, corr.bird.co.k, corr.bird.P11, left=0, right=0)

    # Multipoles for (b1 - b_eta * f * mu^2)^2 P_lin
    # = (b1^2 - 2*b1*b_eta*f*mu^2 + b_eta^2*f^2*mu^4) P_lin
    
    # Transformation to Legendre coefficients:
    # mu^0 = L0
    # mu^2 = 1/3 L0 + 2/3 L2
    # mu^4 = 1/5 L0 + 4/7 L2 + 8/35 L4
    
    # P(k,mu) = [ (b1^2 - 2/3*b1*b_eta*f + 1/5*b_eta^2*f^2) L0
    #           + (-4/3*b1*b_eta*f + 4/7*b_eta^2*f^2) L2
    #           + (8/35*b_eta^2*f^2) L4 ] P_lin
    
    c0 = b1**2 - (2./3.)*b1*b_eta*f + (1./5.)*b_eta**2*f**2
    c2 = -(4./3.)*b1*b_eta*f + (4./7.)*b_eta**2*f**2
    c4 = (8./35.)*b_eta**2*f**2
    
    pk0_p = c0 * pk_lin
    pk2_p = c2 * pk_lin
    pk4_p = c4 * pk_lin

    mu_sq = MU_BIN_CENTERS**2
    L0_p = np.ones_like(MU_BIN_CENTERS)
    L2_p = 0.5 * (3 * mu_sq - 1)
    L4_p = 0.125 * (35 * mu_sq**2 - 30 * mu_sq + 3)

    tree = (pk0_p[:, None] * L0_p[None, :] + 
            pk2_p[:, None] * L2_p[None, :] + 
            pk4_p[:, None] * L4_p[None, :])
    return k_grid, tree


def _load_sherwood_snapshot() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (k_hmpc, mu, p3d_hmpc3, counts) as flat 1-D arrays using direct fitsio read."""
    import fitsio
    p3d_dir = WORKSPACE_ROOT / "data" / "sherwood_p3d" / "data" / "flux_p3d"
    # The simulation is 160 Mpc/h, snapshot 9 (z=2.8).
    files = sorted([f for f in p3d_dir.glob("p3d_160_*.fits") if "_9_" in f.name])
    if not files:
        print(f"Warning: No Sherwood data files found in {p3d_dir}")
        empty = np.array([])
        return empty, empty, empty, empty
    
    # Let's combine all files for this snapshot
    all_k, all_mu, all_p, all_c = [], [], [], []
    for fname in files:
        with fitsio.FITS(str(fname)) as hdul:
            hdu = hdul["FLUX_P3D"]
            p3d = hdu["P3D_HMPC"][:]
            k = hdu["K_HMPC"][:]
            mu = hdu["MU"][:]
            counts = hdu["COUNTS"][:]
            # Only keep non-NaN entries
            mask = ~np.isnan(k) & ~np.isnan(p3d) & ~np.isnan(mu)
            all_k.append(k[mask])
            all_mu.append(mu[mask])
            all_p.append(p3d[mask])
            all_c.append(counts[mask])
    
    return np.concatenate(all_k), np.concatenate(all_mu), np.concatenate(all_p), np.concatenate(all_c)


def _bin_data_in_k(
    k: np.ndarray, p: np.ndarray, counts: np.ndarray, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin (k, p, counts) into k bins; errors from counts via paper_p3d_sigma."""
    edges = np.asarray(bins, dtype=float)
    xc = np.sqrt(edges[:-1] * edges[1:])
    idx = np.digitize(k, edges) - 1
    n_bins = edges.size - 1
    p_med = np.full(n_bins, np.nan)
    sigma = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = idx == i
        if np.count_nonzero(sel) < 1:
            continue
        p_med[i] = float(np.median(p[sel]))
        # paper_p3d_sigma: sqrt(2) * |P| / sqrt(counts), averaged over bin
        p_med[i] = float(np.median(p[sel]))
        c_med = float(np.median(counts[sel]))
        sigma[i] = np.sqrt(2.0) * abs(p_med[i]) / np.sqrt(max(c_med, 1.0))
    keep = np.isfinite(p_med) & np.isfinite(sigma)
    return xc[keep], p_med[keep], sigma[keep]


def _interp_coeffs_at_mu(coeffs: dict[str, np.ndarray], mu: float) -> np.ndarray:
    return (
        np.asarray(coeffs["mu0"]) +
        np.asarray(coeffs["mu2"]) * mu**2 +
        np.asarray(coeffs["mu4"]) * mu**4 +
        np.asarray(coeffs.get("mu6", 0.0)) * mu**6 +
        np.asarray(coeffs.get("mu8", 0.0)) * mu**8
    )


def _coeff_summary(arr: np.ndarray) -> dict[str, float]:
    vals = np.asarray(arr, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {"median_abs": float("nan"), "max_abs": float("nan")}
    return {
        "median_abs": float(np.median(np.abs(finite))),
        "max_abs": float(np.max(np.abs(finite))),
    }


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))

    print("[SHERWOOD-BEFORE-PYBIRD-TREE]", flush=True)
    k_pybird, tree_mu = _build_pybird_tree_spectrum(
        np.asarray(K_GRID, dtype=float),
        float(SHERWOOD_Z28_PARAMS["z"]),
        float(SHERWOOD_Z28_PARAMS["b1"]),
        float(SHERWOOD_Z28_PARAMS["b_eta"]),
    )
    print("[SHERWOOD-AFTER-PYBIRD-TREE]", flush=True)

    cbackend_mu = np.zeros_like(tree_mu)
    op_contrib_mu = np.zeros_like(tree_mu)
    total_mu = tree_mu.copy()
    total_consistency = {pn: np.zeros_like(K_GRID, dtype=float) for pn in ["mu0", "mu2", "mu4", "mu6", "mu8"]}

    sherwood_k, sherwood_mu, sherwood_p, sherwood_counts = _load_sherwood_snapshot()
    normalization_by_mu = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, (ax, mu, label) in enumerate(zip(axes, MU_BIN_CENTERS, MU_BIN_LABELS)):
        lo, hi = float(MU_EDGES[i]), float(MU_EDGES[i + 1])

        y_tree     = K_GRID**2 * tree_mu[:, i]
        y_cbackend = K_GRID**2 * cbackend_mu[:, i]
        y_op       = K_GRID**2 * op_contrib_mu[:, i]
        y_total    = K_GRID**2 * total_mu[:, i]

        ax.axhline(0.0, color="0.8", lw=1.0)
        ax.plot(K_GRID, y_tree,     lw=1.8, ls="--", color="tab:blue",   label=r"tree $(b_1-b_\eta f\mu^2)^2P_{\rm lin}$")
        ax.plot(K_GRID, y_cbackend, lw=1.8, ls="-.", color="tab:green",  label="pybird-only baseline")
        ax.plot(K_GRID, y_op,       lw=1.8, ls=":",  color="tab:orange", label="higher-order terms deferred")
        ax.plot(K_GRID, y_total,    lw=2.4,           color="k",          label="pybird tree")

        if sherwood_k.size:
            mask = (sherwood_mu >= lo) & (sherwood_mu < hi) & (sherwood_k >= 0.03)
            if np.count_nonzero(mask) >= 4:
                k_dat, p_dat, p_sig = _bin_data_in_k(
                    sherwood_k[mask], sherwood_p[mask], sherwood_counts[mask], K_PLOT_BINS
                )
                ax.errorbar(
                    k_dat, k_dat**2 * p_dat, yerr=k_dat**2 * p_sig,
                    fmt="o", ms=4.0, lw=1.0, capsize=2.5, alpha=0.8,
                    color="tab:red",
                    label="Sherwood" if i == 0 else None,
                )

                p_model_on_data = np.interp(np.log(k_dat), np.log(K_GRID), total_mu[:, i])
                valid = (
                    np.isfinite(p_dat) & np.isfinite(p_model_on_data)
                    & (np.abs(p_dat) > 0.0)
                    & (np.sign(p_dat) == np.sign(p_model_on_data))
                )
                if np.any(valid):
                    ratio = p_model_on_data[valid] / p_dat[valid]
                    normalization_by_mu.append({
                        "mu_lo": lo, "mu_hi": hi,
                        "median_model_over_data": float(np.median(ratio)),
                        "p16_model_over_data": float(np.percentile(ratio, 16)),
                        "p84_model_over_data": float(np.percentile(ratio, 84)),
                        "n_binned_points": int(np.count_nonzero(valid)),
                    })

        ax.set_xscale("log")
       # ax.set_yscale("log")
        ax.set_ylim(-0.50, (k_dat**2 * p_dat).max() * 1.0)
        ax.set_xlim(0.03,5.0)
        ax.set_title(label)
        ax.grid(alpha=0.25, which="both")
    
    axes[0].legend(frameon=False, fontsize=10, ncol=2, loc=3)
    axes[2].set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[3].set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0].set_ylabel(r"$k^2 P_F(k,\mu)$")
    axes[2].set_ylabel(r"$k^2 P_F(k,\mu)$")
    fig.suptitle("Sherwood z=2.8 best-fit Ly$\\alpha$ P3D breakdown in $\\mu$ bins")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    fig_path = OUTPUT_DIR / "sherwood_bestfit_k2pk_mu_bins.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    np.savez(
        OUTPUT_DIR / "sherwood_bestfit_k2pk_mu_bins.npz",
        k_hmpc=K_GRID,
        mu_edges=MU_EDGES,
        mu_bin_centers=MU_BIN_CENTERS,
        tree=tree_mu,
        cbackend_b1_beta=cbackend_mu,
        operators=op_contrib_mu,
        total=total_mu,
        sherwood_k_hmpc=sherwood_k,
        sherwood_mu=sherwood_mu,
        sherwood_p3d_hmpc3=sherwood_p,
        sherwood_counts=sherwood_counts,
    )

    summary = {
        "figure": str(fig_path),
        "z": SHERWOOD_Z28_PARAMS["z"],
        "mu_edges": MU_EDGES.tolist(),
        "mu_bin_centers": MU_BIN_CENTERS.tolist(),
        "k_min": float(K_GRID.min()),
        "k_max": float(K_GRID.max()),
        "normalization_diagnostic": normalization_by_mu,
        "backend_total_vs_python_assembled": {
            pn: {
                **_coeff_summary(total_consistency[pn]),
                "signed_at_kmin": float(np.asarray(total_consistency[pn])[0]),
                "signed_at_kmax": float(np.asarray(total_consistency[pn])[-1]),
            }
            for pn in ["mu0", "mu2", "mu4", "mu6", "mu8"]
        },
        "params": SHERWOOD_Z28_PARAMS,
    }
    (OUTPUT_DIR / "sherwood_bestfit_k2pk_mu_bins.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if normalization_by_mu:
        print("Normalization diagnostic (median model/data by mu bin):")
        for row in normalization_by_mu:
            print(
                f"  mu=[{row['mu_lo']:.2f},{row['mu_hi']:.2f}): "
                f"median={row['median_model_over_data']:.4f} "
                f"[p16={row['p16_model_over_data']:.4f}, p84={row['p84_model_over_data']:.4f}]"
            )

    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
