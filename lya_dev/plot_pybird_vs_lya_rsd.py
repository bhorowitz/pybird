#!/usr/bin/env python3
"""Compare PyBird original 1-loop RSD vs LyaNonLinear (no PROJ) in 4 mu bins.

PyBird:  b1=1, all other bias/EFT=0, standard galaxy RSD code.
LyaNon:  b1=1, b_eta=-1  →  beta = b_eta*f = -f, PROJ channels zeroed out.

At tree level both give  (1 + f*mu^2)^2 * P_lin  (Kaiser formula),
so any difference is purely in the 1-loop terms.

Output: lya-dev/output/pybird_vs_lya_rsd_mu_bins.png
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

BASELINE_COSMO = {
    "h": 0.678,
    "omega_b": 0.0482 * 0.678**2,
    "omega_cdm": (0.308 - 0.0482) * 0.678**2,
    "ln10^{10}A_s": 3.044,
    "n_s": 0.961,
}
Z = 2.8

MU_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.01])
MU_BIN_CENTERS = 0.5 * (MU_EDGES[:-1] + MU_EDGES[1:])
MU_BIN_LABELS = [
    r"$\mu \in [0.00,\,0.25)$",
    r"$\mu \in [0.25,\,0.50)$",
    r"$\mu \in [0.50,\,0.75)$",
    r"$\mu \in [0.75,\,1.00)$",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legendre(ell: int, mu: np.ndarray) -> np.ndarray:
    if ell == 0:
        return np.ones_like(mu)
    if ell == 2:
        return 0.5 * (3 * mu**2 - 1)
    if ell == 4:
        return 0.125 * (35 * mu**4 - 30 * mu**2 + 3)
    raise ValueError(ell)


def multipoles_to_p_at_mu(multipoles: np.ndarray, mu: float) -> np.ndarray:
    """Reconstruct P(k, mu) from shape-(3, Nk) multipoles (ell=0,2,4)."""
    return (multipoles[0] * _legendre(0, mu)
            + multipoles[1] * _legendre(2, mu)
            + multipoles[2] * _legendre(4, mu))


# ---------------------------------------------------------------------------
# PyBird 1-loop power spectrum
# ---------------------------------------------------------------------------

def run_pybird(k_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (P_tree, P_fullloop) each shape (4, Nk) at MU_BIN_CENTERS, and f."""
    from pybird.correlator import Correlator
    from pybird.cosmo import Cosmo

    corr = Correlator()
    corr.set({
        "output": "bPk", "z": Z,
        "with_bias": True, "with_resum": False, "with_time": False,
        "with_exact_time": False, "with_redshift_bin": False,
        "with_ap": False, "with_binning": False,
        "with_survey_mask": False, "with_fibercol": False,
        "kmin": 5e-4, "kmax": 0.6, "multipole": 3,
        "eft_basis": "eftoflss", "km": 1.0, "kr": 1.0,
    })
    cosmo = Cosmo(corr.c)
    cd = cosmo.set_cosmo(BASELINE_COSMO, module="Symbolic")
    cd["bias"] = {
        "b1": 1.0, "b2": 0.0, "b3": 0.0, "b4": 0.0,
        "cct": 0.0, "cr1": 0.0, "cr2": 0.0,
    }
    corr.compute(cosmo_dict=cd, do_core=True, do_survey_specific=False)
    bird = corr.bird
    f = float(bird.f)

    k_bird = bird.co.k          # (Nk_bird,)
    # Ps: shape (2, 3, Nk_bird) — axis0: [tree, loop], axis1: [l=0,2,4]
    tree_multi = bird.Ps[0]     # (3, Nk_bird)
    bird.setfullPs()
    full_multi = bird.fullPs    # (3, Nk_bird) — tree+loop

    # Interpolate to k_out and evaluate at each mu bin center
    P_tree = np.zeros((len(MU_BIN_CENTERS), len(k_out)))
    P_full = np.zeros((len(MU_BIN_CENTERS), len(k_out)))
    for mi, mu_c in enumerate(MU_BIN_CENTERS):
        for li, ell in enumerate([0, 2, 4]):
            leg = float(_legendre(ell, mu_c))
            P_tree[mi] += np.interp(k_out, k_bird, tree_multi[li]) * leg
            P_full[mi] += np.interp(k_out, k_bird, full_multi[li]) * leg

    return P_tree, P_full, f


# ---------------------------------------------------------------------------
# LyaNonLinear 1-loop power spectrum (no PROJ)
# ---------------------------------------------------------------------------

class _NoProjMatrices:
    """Wrapper that zeros out PROJ T13 channels (leaves F3/G3 and all T22)."""
    def __init__(self, m):
        self._t13_radial = m._t13_radial
        self._t13_mat    = m._t13_mat
        self._t13_proj   = {}        # PROJ channels zeroed
        self._t22_mat    = m._t22_mat
        self.nmax         = m.nmax
        self.fftlog_params = m.fftlog_params

    # forward accessors needed by LyaNonLinear
    def get_t13_radial(self, ch):  return self._t13_radial[ch]
    def t13_mu_k_powers(self, ch): return []
    def t13_proj_mu_k_powers(self, ch): return []
    def get_t22_matrix(self, qi, qj, mu, kp):
        return self._t22_mat[(qi, qj)][mu][kp]
    def t22_mu_k_powers(self, qi, qj):
        return sorted(
            (mu, kp)
            for mu, kd in self._t22_mat.get((qi, qj), {}).items()
            for kp in kd
        )


def run_lya(k_out: np.ndarray, f: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_tree, P_full) each shape (4, Nk) at MU_BIN_CENTERS.

    b1=1, b_eta=-1  →  beta = b_eta*f = -f.  PROJ channels zeroed.
    """
    from pybird.fftlog import FFTLog
    from pybird.lya_common import LyaCommon, MU_POWERS, BASIS_LABELS
    from pybird.lya_matrices import LyaMatrices
    from pybird.lya_nonlinear import LyaNonLinear

    # Load P_lin from CLASS cache
    class_cache = OUTPUT_DIR / "class_plin_z28.npz"
    d = np.load(class_cache)
    pk_lin = np.interp(k_out, d["kk"], d["pk"], left=0.0, right=0.0)

    matrices = LyaMatrices(MANIFEST_128)
    no_proj  = _NoProjMatrices(matrices)
    fp       = matrices.fftlog_params

    common     = LyaCommon(kmin=float(k_out.min()), kmax=float(k_out.max()), Nk=len(k_out))
    common.k   = k_out
    fft        = FFTLog(Nmax=fp["nmax"], xmin=fp["k0"], xmax=fp["kmax_fft"], bias=fp["fftlog_bias"])

    nl = LyaNonLinear(common, no_proj, fft)
    P13, P22 = nl.make_loop(pk_lin)    # (N_basis, N_mu_pow, Nk)

    b1   = 1.0
    beta = f           # b_eta=-1 → beta = b_eta*f = -f
    # basis: ["1","b1","b1^2","beta","b1_beta","beta^2"]
    bias_vec = np.array([1.0, b1, b1**2, beta, b1*beta, beta**2])

    # P(k, mu) = sum_b bias[b] * sum_{mu_pow_idx} P[b, mu_pow_idx, k] * mu^{pow}
    P_tree = np.zeros((len(MU_BIN_CENTERS), len(k_out)))
    P_loop = np.zeros((len(MU_BIN_CENTERS), len(k_out)))

    mu_pows = np.array(MU_POWERS)

    for mi, mu_c in enumerate(MU_BIN_CENTERS):
        mu_factors = mu_c ** mu_pows    # (5,)
        for bi, coeff in enumerate(bias_vec):
            for pi, mu_f in enumerate(mu_factors):
                P_loop[mi] += coeff * (P13[bi, pi] + P22[bi, pi]) * mu_f

        # tree: (b1 + f*mu^2)^2 * P_lin  [since b_eta=-1 → b1 - b_eta*f*mu^2 = b1+f*mu^2]
        P_tree[mi] = (b1 + f * mu_c**2)**2 * pk_lin

    P_full = P_tree + P_loop
    return P_tree, P_full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # k range safe for both codes
    K_GRID = np.geomspace(5e-3, 0.5, 200)

    print("Running PyBird …")
    pb_tree, pb_full, f = run_pybird(K_GRID)
    print(f"  f = {f:.6f}")

    print("Running LyaNonLinear (no PROJ) …")
    lya_tree, lya_full = run_lya(K_GRID, f)

    # Load CLASS P_lin for context
    d = np.load(OUTPUT_DIR / "class_plin_z28.npz")
    pk_lin = np.interp(K_GRID, d["kk"], d["pk"], left=0.0, right=0.0)

    # -------------------------------------------------------------------
    # Figure 1 — absolute P(k,mu) comparison
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    for i, (ax, mu_c, lbl) in enumerate(zip(axes, MU_BIN_CENTERS, MU_BIN_LABELS)):
        w = K_GRID**2
        tree_ref = (1 + f * mu_c**2)**2 * pk_lin  # Kaiser tree

        ax.plot(K_GRID, w * tree_ref,     color="0.65", lw=1.2, ls="--", label="Kaiser tree" if i==0 else "_")
        ax.plot(K_GRID, w * pb_full[i],   color="tab:blue",   lw=2.0, label="PyBird b1=1" if i==0 else "_")
        ax.plot(K_GRID, w * lya_full[i],  color="tab:orange", lw=1.8, ls="--", label="LyaNon b1=1, b_eta=-1 (no PROJ)" if i==0 else "_")

        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(5e-3, 0.5)
        ax.grid(alpha=0.2, which="both")
        ax.set_title(lbl, fontsize=11)

    axes[0].legend(frameon=False, fontsize=10, ncol=1, loc="upper left")
    for ax in [axes[2], axes[3]]:
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    for ax in [axes[0], axes[2]]:
        ax.set_ylabel(r"$k^2 P(k,\mu)$")
    fig.suptitle(
        "1-loop  P(k,μ):  PyBird  vs  LyaNonLinear (no PROJ)\n"
        f"z={Z},  b1=1,  b_eta=-1,  f={f:.4f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out1 = OUTPUT_DIR / "pybird_vs_lya_rsd_mu_bins.png"
    fig.savefig(out1, dpi=180)
    plt.close(fig)
    print(f"Wrote {out1}")

    # -------------------------------------------------------------------
    # Figure 2 — residuals  (LyaNon - PyBird) / PyBird
    # -------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes2 = axes2.ravel()

    for i, (ax, mu_c, lbl) in enumerate(zip(axes2, MU_BIN_CENTERS, MU_BIN_LABELS)):
        denom = np.abs(pb_full[i]) + 1e-30
        loop_diff = lya_full[i] - pb_full[i]
        rel = loop_diff / denom

        ax.plot(K_GRID, rel, color="tab:purple", lw=1.8,
                label=r"$\Delta P / |P_{\rm PyBird}|$" if i==0 else "_")
        ax.axhline(0, color="0.6", lw=1.0)
        ax.axhline( 0.01, color="0.8", lw=0.8, ls=":")
        ax.axhline(-0.01, color="0.8", lw=0.8, ls=":")

        ax.set_xscale("log")
        ax.set_xlim(5e-3, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.grid(alpha=0.2, which="both")
        ax.set_title(lbl, fontsize=11)

    axes2[0].legend(frameon=False, fontsize=10)
    for ax in [axes2[2], axes2[3]]:
        ax.set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    for ax in [axes2[0], axes2[2]]:
        ax.set_ylabel(r"$(P_{\rm Lya} - P_{\rm PyBird})\,/\,|P_{\rm PyBird}|$")
    fig2.suptitle(
        "Residual:  LyaNonLinear (no PROJ) − PyBird\n"
        f"z={Z},  b1=1,  b_eta=-1,  f={f:.4f}",
        fontsize=12,
    )
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = OUTPUT_DIR / "pybird_vs_lya_rsd_residual.png"
    fig2.savefig(out2, dpi=180)
    plt.close(fig2)
    print(f"Wrote {out2}")

    # quick numerical summary
    print("\nMax |relative residual| per mu bin (k < 0.3 h/Mpc):")
    mask = K_GRID < 0.3
    for i, mu_c in enumerate(MU_BIN_CENTERS):
        rel = np.abs(lya_full[i][mask] - pb_full[i][mask]) / (np.abs(pb_full[i][mask]) + 1e-30)
        print(f"  mu~{mu_c:.3f}:  max={rel.max():.4f},  median={np.median(rel):.4f}")


if __name__ == "__main__":
    main()
