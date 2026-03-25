#!/usr/bin/env python3
"""Plot individual bias-term contributions to the Sherwood z=2.8 LyA P3D.

Loads the CLASS P_lin cache and Sherwood data produced by
plot_sherwood_bestfit_mu_bins.py, then plots each term of

  P(k,mu) = P_tree + P_loop

where P_tree breaks down as:
  b1^2 * P_lin          (mu^0 tree)
  -2*b1*b_eta*f*mu^2 * P_lin   (mu^2 tree cross)
  b_eta^2*f^2*mu^4 * P_lin     (mu^4 tree)

and P_loop adds the one-loop contributions from LyaNonLinear
(T13 + T22 per reduced basis label).

Output: lya-dev/output/sherwood_term_breakdown.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "lya-dev" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_128 = ROOT / "compute_matrices_python" / "lya_generated_tables" / "lya_reduced_manifest_n128.json"

SHERWOOD_Z28_PARAMS = {
    "z": 2.8,
    "b1": -0.24168,
    "b_eta": 0.324,
}

MU_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.01])
MU_BIN_CENTERS = 0.5 * (MU_EDGES[:-1] + MU_EDGES[1:])
MU_BIN_LABELS = [
    r"$\mu \in [0.00,\,0.25)$",
    r"$\mu \in [0.25,\,0.50)$",
    r"$\mu \in [0.50,\,0.75)$",
    r"$\mu \in [0.75,\,1.00)$",
]

K_PLOT_BINS = np.logspace(np.log10(0.04), np.log10(3.0), 14)


# ---------------------------------------------------------------------------
# Load CLASS P_lin and cosmological quantities
# ---------------------------------------------------------------------------

def _load_class_cache():
    cache = OUTPUT_DIR / "class_plin_z28.npz"
    if not cache.exists():
        raise FileNotFoundError(f"Run plot_sherwood_bestfit_mu_bins.py first to create {cache}")
    d = np.load(cache)
    return d["kk"], d["pk"], float(d["f"])


# ---------------------------------------------------------------------------
# Sherwood data
# ---------------------------------------------------------------------------

def _load_sherwood():
    try:
        import fitsio
    except ImportError:
        return np.array([]), np.array([]), np.array([]), np.array([])
    p3d_dir = ROOT / "data" / "sherwood_p3d" / "data" / "flux_p3d"
    files = sorted([f for f in p3d_dir.glob("p3d_160_*.fits") if "_9_" in f.name])
    if not files:
        return np.array([]), np.array([]), np.array([]), np.array([])
    all_k, all_mu, all_p, all_c = [], [], [], []
    for fname in files:
        with fitsio.FITS(str(fname)) as hdul:
            hdu = hdul["FLUX_P3D"]
            k = hdu["K_HMPC"][:]
            mu = hdu["MU"][:]
            p3d = hdu["P3D_HMPC"][:]
            counts = hdu["COUNTS"][:]
            mask = ~np.isnan(k) & ~np.isnan(p3d) & ~np.isnan(mu)
            all_k.append(k[mask]); all_mu.append(mu[mask])
            all_p.append(p3d[mask]); all_c.append(counts[mask])
    return (np.concatenate(all_k), np.concatenate(all_mu),
            np.concatenate(all_p), np.concatenate(all_c))


def _bin_data_in_k(k, p, counts, bins):
    edges = np.asarray(bins)
    xc = np.sqrt(edges[:-1] * edges[1:])
    idx = np.digitize(k, edges) - 1
    n = edges.size - 1
    p_med = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    for i in range(n):
        sel = idx == i
        if np.count_nonzero(sel) < 1:
            continue
        p_med[i] = float(np.median(p[sel]))
        c_med = float(np.median(counts[sel]))
        sigma[i] = np.sqrt(2.0) * abs(p_med[i]) / np.sqrt(max(c_med, 1.0))
    keep = np.isfinite(p_med) & np.isfinite(sigma)
    return xc[keep], p_med[keep], sigma[keep]


# ---------------------------------------------------------------------------
# LyaNonLinear loop terms
# ---------------------------------------------------------------------------

def _compute_loop_terms(k_grid, pk_lin_on_grid, f):
    """Return dict {basis_label: P(k, mu_idx)} shape (4,) per k.

    Returns P13+P22 summed over sectors, evaluated at each mu bin center.
    """
    if not MANIFEST_128.exists():
        print(f"  Loop tables not found at {MANIFEST_128}, skipping loop terms.")
        return {}

    from pybird.fftlog import FFTLog
    from pybird.lya_common import LyaCommon, MU_POWERS, BASIS_LABELS
    from pybird.lya_matrices import LyaMatrices
    from pybird.lya_nonlinear import LyaNonLinear

    matrices = LyaMatrices(MANIFEST_128)
    fp = matrices.fftlog_params
    Nk = len(k_grid)
    common = LyaCommon(kmin=float(k_grid.min()), kmax=float(k_grid.max()), Nk=Nk)
    # Align k_grid precisely
    common.k = k_grid

    fft = FFTLog(
        Nmax=fp["nmax"],
        xmin=fp["k0"],
        xmax=fp["kmax_fft"],
        bias=fp["fftlog_bias"],
    )

    nl = LyaNonLinear(common, matrices, fft)
    P13, P22 = nl.make_loop(pk_lin_on_grid)   # (N_basis, N_mu, Nk)

    # Evaluate at mu bin centers (mu^0..mu^8 power array → bin values)
    # MU_POWERS = [0, 2, 4, 6, 8], shape (5,)
    # P13/P22 axis 1 is mu_power index (0→mu^0, 1→mu^2, ...)
    mu_pows = np.array(MU_POWERS)   # [0, 2, 4, 6, 8]

    loop = {}
    for bi, label in enumerate(BASIS_LABELS):
        # shape (N_mu_powers, Nk) for this basis label
        p13_b = P13[bi]  # (5, Nk)
        p22_b = P22[bi]  # (5, Nk)
        p_total_b = p13_b + p22_b  # (5, Nk)
        # evaluate at each mu bin center: sum_i P_mu[i] * mu^pow_i
        vals = np.zeros((len(MU_BIN_CENTERS), Nk))
        for mi, mu_c in enumerate(MU_BIN_CENTERS):
            for pi, pw in enumerate(mu_pows):
                vals[mi] += p_total_b[pi] * (mu_c ** pw)
        loop[label] = vals   # (4, Nk)
    return loop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    kk_class, pk_class, f = _load_class_cache()
    b1    = SHERWOOD_Z28_PARAMS["b1"]
    b_eta = SHERWOOD_Z28_PARAMS["b_eta"]
    beta  = b_eta * f    # = b_eta * f in the reduced basis

    # Fine k grid for smooth theory curves
    K_GRID = np.geomspace(1e-2, 5.0, 240)
    pk_lin = np.interp(K_GRID, kk_class, pk_class, left=0.0, right=0.0)

    # ---------------------------------------------------------------------------
    # Tree-level terms at each mu bin
    # ---------------------------------------------------------------------------
    # P_tree(k,mu) = (b1 - b_eta * f * mu^2)^2 * P_lin
    #              = b1^2 * P_lin  +  (-2*b1*b_eta*f) * mu^2 * P_lin  +  (b_eta^2*f^2) * mu^4 * P_lin

    coeff_b1sq   = b1**2                   # ~ +0.058
    coeff_cross  = -2.0 * b1 * b_eta * f   # ~ +0.153 (b1 negative)
    coeff_beta2  = (b_eta * f)**2          # ~ +0.097

    tree_terms = {}
    for mu_c in MU_BIN_CENTERS:
        pass  # computed per-bin below

    tree_b1sq  = coeff_b1sq  * pk_lin           # (Nk,) — no mu dep
    tree_cross = coeff_cross * pk_lin            # (Nk,) — to be * mu^2
    tree_beta2 = coeff_beta2 * pk_lin            # (Nk,) — to be * mu^4
    tree_total = np.zeros((len(K_GRID), 4))
    tree_b1sq_mu = np.zeros((len(K_GRID), 4))
    tree_cross_mu = np.zeros((len(K_GRID), 4))
    tree_beta2_mu = np.zeros((len(K_GRID), 4))
    for mi, mu_c in enumerate(MU_BIN_CENTERS):
        tree_b1sq_mu[:, mi]  = tree_b1sq
        tree_cross_mu[:, mi] = tree_cross * mu_c**2
        tree_beta2_mu[:, mi] = tree_beta2 * mu_c**4
        tree_total[:, mi]    = tree_b1sq_mu[:, mi] + tree_cross_mu[:, mi] + tree_beta2_mu[:, mi]

    # ---------------------------------------------------------------------------
    # Loop terms
    # ---------------------------------------------------------------------------
    print("Computing loop terms …")
    loop = _compute_loop_terms(K_GRID, pk_lin, f)

    if loop:
        # Combine loop terms by their physical bias coefficient
        # basis = ["1", "b1", "b1^2", "beta", "b1_beta", "beta^2"]
        bias_coeffs = {
            "1":        1.0,
            "b1":       b1,
            "b1^2":     b1**2,
            "beta":     beta,
            "b1_beta":  b1 * beta,
            "beta^2":   beta**2,
        }
        loop_total = np.zeros((4, len(K_GRID)))
        for label, coeff in bias_coeffs.items():
            if label in loop:
                loop_total += coeff * loop[label]
        loop_total = loop_total.T   # (Nk, 4)
    else:
        loop_total = np.zeros_like(tree_total)

    total = tree_total + loop_total

    # ---------------------------------------------------------------------------
    # Sherwood data
    # ---------------------------------------------------------------------------
    sh_k, sh_mu, sh_p, sh_c = _load_sherwood()

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    COLORS = {
        "tree_b1sq":   "tab:blue",
        "tree_cross":  "tab:orange",
        "tree_beta2":  "tab:green",
        "tree_total":  "k",
        "loop_total":  "tab:purple",
        "grand_total": "tab:red",
    }

    for i, (ax, mu_c, label) in enumerate(zip(axes, MU_BIN_CENTERS, MU_BIN_LABELS)):
        lo, hi = float(MU_EDGES[i]), float(MU_EDGES[i + 1])
        w = K_GRID**2

        # Sherwood data
        if sh_k.size:
            mask = (sh_mu >= lo) & (sh_mu < hi) & (sh_k >= 0.03)
            if np.count_nonzero(mask) >= 4:
                k_d, p_d, p_sig = _bin_data_in_k(sh_k[mask], sh_p[mask], sh_c[mask], K_PLOT_BINS)
                ax.errorbar(k_d, k_d**2 * p_d, yerr=k_d**2 * p_sig,
                            fmt="o", ms=4, lw=1, capsize=2.5, alpha=0.85,
                            color="0.3", label="Sherwood" if i == 0 else "_")

        # Tree-level terms
        ax.plot(K_GRID, w * tree_b1sq_mu[:, i],  lw=1.5, ls="--",
                color=COLORS["tree_b1sq"],
                label=r"$b_1^2\,P_{\rm lin}$" if i == 0 else "_")
        ax.plot(K_GRID, w * tree_cross_mu[:, i], lw=1.5, ls="-.",
                color=COLORS["tree_cross"],
                label=r"$-2b_1 b_\eta f\mu^2\,P_{\rm lin}$" if i == 0 else "_")
        ax.plot(K_GRID, w * tree_beta2_mu[:, i], lw=1.5, ls=":",
                color=COLORS["tree_beta2"],
                label=r"$b_\eta^2 f^2\mu^4\,P_{\rm lin}$" if i == 0 else "_")
        ax.semilogy(K_GRID, w * tree_total[:, i],    lw=2.0, ls="-",
                color=COLORS["tree_total"],
                label="tree total" if i == 0 else "_")

        # Loop contributions
        if np.any(loop_total != 0):
            ax.plot(K_GRID, w * loop_total[:, i], lw=1.8, ls=(0, (3,1,1,1)),
                    color=COLORS["loop_total"],
                    label="loop total" if i == 0 else "_")
            ax.plot(K_GRID, w * total[:, i], lw=2.4, ls="-",
                    color=COLORS["grand_total"],
                    label="tree + loop" if i == 0 else "_")

        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.03, 5.0)
        ax.set_ylim(1e-1, 1e1)
        ax.grid(alpha=0.2, which="both")
        ax.set_title(label, fontsize=11)

        # mu-dependent coefficients for this panel
        mu_info = (
            f"$b_1^2={coeff_b1sq:.4f}$\n"
            f"$-2b_1 b_\\eta f\\mu^2={coeff_cross * mu_c**2:.4f}$\n"
            f"$b_\\eta^2 f^2 \\mu^4={coeff_beta2 * mu_c**4:.4f}$"
        )
        ax.text(0.97, 0.97, mu_info, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.7))

    axes[0].legend(frameon=False, fontsize=9, ncol=2, loc="lower left")
    for ax in [axes[2], axes[3]]:
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    for ax in [axes[0], axes[2]]:
        ax.set_ylabel(r"$k^2\,P_F(k,\mu)\ [({\rm Mpc}/h)^{-1}]$")

    param_str = (
        f"z={SHERWOOD_Z28_PARAMS['z']},  "
        f"$b_1={b1}$,  "
        f"$b_{{\\eta}}={b_eta}$,  "
        f"$f={f:.4f}$,  "
        f"$\\beta=b_{{\\eta}}f={beta:.4f}$"
    )
    fig.suptitle(
        "Sherwood z=2.8  —  bias-term breakdown\n" + param_str,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = OUTPUT_DIR / "sherwood_term_breakdown.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
