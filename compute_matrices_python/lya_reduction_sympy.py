#!/usr/bin/env python3
"""
lya_reduction.py  –  Symbolic reduction engine for Lyman-alpha one-loop integrands
==================================================================================

Implements steps A1–A2 of the LyA matrix-generation guide.

For the 22-sector:
    Reduces every  Q_i(q, k−q) × Q_j(q, k−q)  product to a sparse list
    of canonical master-integral contributions:

        coeff × k^m × μ^ℓ_explicit × ∫ (ẑ·q)^r / [q^{2(ν₁+Δν₁)} |k−q|^{2(ν₂+Δν₂)}]

    The integral is then decomposed via the known An, Bn, … master components
    (Eq. 3.23 and Appendix A of Ivanov 2309.10133) to produce final
    contributions to each  μ^{2n}  coefficient  P_n^{(22)}.

For the 13-sector:
    Provides analogous reductions for each T13_B channel from Eq. (3.20).

Output is a JSON descriptor per (channel, mu_power) pair, following the
schema in Section D of the guide.

References
----------
Paper : Ivanov, 2309.10133v2  (Eqs. 3.19, 3.20, 3.23, 3.24, App. A)
Guide : lya_matrix_generation_guide.txt
"""

import sympy as sp
from sympy import Rational as R, Integer, symbols, expand, Poly
from collections import defaultdict
import json
import os
import sys
import time

from lya_channels import T13_CHANNEL_MAP

# ════════════════════════════════════════════════════════════════
#  §1  SYMBOLIC VARIABLES
# ════════════════════════════════════════════════════════════════

k   = symbols('k',   positive=True)   # external momentum  |k|
q   = symbols('q',   positive=True)   # loop momentum      |q|
kmq = symbols('kmq', positive=True)   # loop momentum      |k − q|
mu  = symbols('mu')                   # ẑ · k̂   (external LOS cosine)
zq  = symbols('zq')                   # ẑ · q   (loop LOS projection)

# Derived composite objects (always substituted before expansion)
# ẑ · (k − q)  =  k μ  −  ẑ · q
zkq = k * mu - zq

# q · (k − q)  =  (k² − q² − |k−q|²) / 2
qdotkmq = (k**2 - q**2 - kmq**2) / 2


# ════════════════════════════════════════════════════════════════
#  §2  SPT KERNEL NUMERATORS (×q²|k−q|² to clear denominators)
# ════════════════════════════════════════════════════════════════

def F2_numer():
    r"""Numerator of  F₂(q, k−q) × q² × |k−q|²

    F₂(k₁,k₂) = 5/7 + (k₁·k₂)/(2k₁k₂)(k₁/k₂+k₂/k₁) + 2/7 [(k₁·k₂)/(k₁k₂)]²
    """
    return (R(5, 7) * q**2 * kmq**2
            + qdotkmq * (q**2 + kmq**2) / 2
            + R(2, 7) * qdotkmq**2)


def G2_numer():
    r"""Numerator of  G₂(q, k−q) × q² × |k−q|²

    G₂(k₁,k₂) = 3/7 + (k₁·k₂)/(2k₁k₂)(k₁/k₂+k₂/k₁) + 4/7 [(k₁·k₂)/(k₁k₂)]²
    """
    return (R(3, 7) * q**2 * kmq**2
            + qdotkmq * (q**2 + kmq**2) / 2
            + R(4, 7) * qdotkmq**2)


# ════════════════════════════════════════════════════════════════
#  §3  PRIMITIVE Q_i BASIS  (22-sector)
# ════════════════════════════════════════════════════════════════
#
# Each function returns  (P, dq, dkmq)  where
#     Q_i  =  P  /  (q^dq  ×  |k−q|^dkmq)
# and P is a polynomial in {k, μ, ẑ·q, q², |k−q|²}.
#
# Convention: q and |k−q| appear only as q² and kmq² inside P.
# This is verified by the monomial collector.

def qi_DELTA2():
    """1   (constant – carries b₂/2)"""
    return (Integer(1), 0, 0)

def qi_G2():
    """μ₁₂² − 1  =  [(q·(k−q))² − q²|k−q|²] / (q²|k−q|²)"""
    P = qdotkmq**2 - q**2 * kmq**2
    return (expand(P), 2, 2)

def qi_F2():
    """F₂(q, k−q)   (carries b₁)"""
    return (expand(F2_numer()), 2, 2)

def qi_G2_MU():
    """−μ² G₂(q, k−q)   (carries bη f)"""
    return (expand(-mu**2 * G2_numer()), 2, 2)

def qi_DELTAETA():
    """−(μ₁² + μ₂²)/2  =  −[zq² kmq² + (kμ−zq)² q²] / (2 q² kmq²)

    Carries  f bδη.
    """
    P = -(zq**2 * kmq**2 + zkq**2 * q**2) / 2
    return (expand(P), 2, 2)

def qi_ETA2():
    """μ₁² μ₂²  =  zq² (kμ−zq)² / (q² kmq²)

    Carries  f² bη².
    """
    P = zq**2 * zkq**2
    return (expand(P), 2, 2)

def qi_RSD_B1():
    """(μ₁μ₂/2)(k₂/k₁ + k₁/k₂)  =  zq(kμ−zq)(q²+kmq²) / (2 q² kmq²)

    Carries  b₁ f.
    """
    P = zq * zkq * (q**2 + kmq**2) / 2
    return (expand(P), 2, 2)

def qi_RSD_BETA():
    """−(μ₁μ₂/2)(k₂μ₂²/k₁ + k₁μ₁²/k₂)  =  −zq(kμ−zq)(zkq²+zq²) / (2 q² kmq²)

    Carries  bη f².
    """
    P = -zq * zkq * (zkq**2 + zq**2) / 2
    return (expand(P), 2, 2)

def qi_KKPAR():
    r"""μ₁ μ₂ μ₁₂ − (μ₁²+μ₂²)/3 + 1/9

    The traceless LOS tensor piece.  Carries b_{(KK)∥}.
    """
    P = (zq * zkq * qdotkmq
         - (zq**2 * kmq**2 + zkq**2 * q**2) / 3
         + q**2 * kmq**2 / 9)
    return (expand(P), 2, 2)

def qi_PI2PAR():
    r"""μ₁ μ₂ μ₁₂ + (5/7) μ² (1 − μ₁₂²)

    The Π^[2]_∥ piece.  Carries b_{Π^[2]_∥}.
    """
    P = (zq * zkq * qdotkmq
         + R(5, 7) * mu**2 * (q**2 * kmq**2 - qdotkmq**2))
    return (expand(P), 2, 2)


# — Registry —
Q_BASIS = {
    'DELTA2':   qi_DELTA2,
    'G2':       qi_G2,
    'F2':       qi_F2,
    'G2_MU':    qi_G2_MU,
    'DELTAETA': qi_DELTAETA,
    'ETA2':     qi_ETA2,
    'RSD_B1':   qi_RSD_B1,
    'RSD_BETA': qi_RSD_BETA,
    'KKPAR':    qi_KKPAR,
    'PI2PAR':   qi_PI2PAR,
}
Q_NAMES = list(Q_BASIS.keys())
NQ = len(Q_NAMES)


# ════════════════════════════════════════════════════════════════
#  §4  MASTER-INTEGRAL COMPONENT LOOKUP
# ════════════════════════════════════════════════════════════════
#
# For LOS rank r, the master integral  ∫ (ẑ·q)^r / [q^{2ν₁} |k−q|^{2ν₂}]
# decomposes as  k^{3−2ν₁₂+r} × (polynomial in μ).
#
# Even r:  k^r (A_r + μ² B_r + μ⁴ C_r + …)
# Odd  r:  k^r μ (A_r + μ² B_r + μ⁴ C_r + …)
#
# Each component adds a known implicit μ power.
# For r ≤ 4 the components exist in Ref. [94] (CLASS-PT).
# For r ≥ 5 they are defined in Appendix A of the paper.

MASTER_COMPONENTS = {
    0: [('J',  0)],
    1: [('A1', 1)],
    2: [('A2', 0), ('B2', 2)],
    3: [('A3', 1), ('B3', 3)],
    4: [('A4', 0), ('B4', 2), ('C4', 4)],
    5: [('A5', 1), ('B5', 3), ('C5', 5)],
    6: [('A6', 0), ('B6', 2), ('C6', 4), ('D6', 6)],
    7: [('A7', 1), ('B7', 3), ('C7', 5), ('D7', 7)],
    8: [('A8', 0), ('B8', 2), ('C8', 4), ('D8', 6), ('E8', 8)],
}


# ════════════════════════════════════════════════════════════════
#  §5  MONOMIAL COLLECTION
# ════════════════════════════════════════════════════════════════

def collect_monomials(expr):
    """Decompose a polynomial in {k, μ, zq, q, kmq} into monomials.

    Returns
    -------
    dict :  (a_k, b_mu, r_zq, p_q2, s_kmq2) -> Rational coefficient
        where  p_q2 = (power of q) / 2,  s_kmq2 = (power of kmq) / 2.
        Raises ValueError if odd q or kmq powers appear.
    """
    expr_exp = expand(expr)
    if expr_exp == 0:
        return {}

    p = Poly(expr_exp, k, mu, zq, q, kmq, domain='QQ')

    result = defaultdict(lambda: R(0))
    for monom, coeff in p.as_dict().items():
        a_k, b_mu, r_zq, p_q, s_kmq = monom

        if p_q % 2 != 0:
            raise ValueError(
                f"Odd q-power {p_q} in term "
                f"k^{a_k} mu^{b_mu} zq^{r_zq} q^{p_q} kmq^{s_kmq}")
        if s_kmq % 2 != 0:
            raise ValueError(
                f"Odd kmq-power {s_kmq} in term "
                f"k^{a_k} mu^{b_mu} zq^{r_zq} q^{p_q} kmq^{s_kmq}")

        key = (a_k, b_mu, r_zq, p_q // 2, s_kmq // 2)
        result[key] += R(coeff)

    return {kk: v for kk, v in result.items() if v != 0}


# ════════════════════════════════════════════════════════════════
#  §6  22-SECTOR PAIR REDUCTION  (raw monomials)
# ════════════════════════════════════════════════════════════════

def reduce_22_pair_raw(name_i, name_j):
    """Reduce Q_i x Q_j to raw monomial form.

    Returns
    -------
    list of dicts, each with keys:
        coeff        : sympy Rational
        explicit_mu  : int (power of mu from the kernel expansion)
        los_rank_r   : int (power of zq, determines master integral family)
        delta_nu1    : int (Mellin shift on nu_1)
        delta_nu2    : int (Mellin shift on nu_2)
        extra_k_power: int (k power from kernel beyond master integral's own)
    """
    Pi, dqi, dkmqi = Q_BASIS[name_i]()
    Pj, dqj, dkmqj = Q_BASIS[name_j]()

    total_dq   = dqi  + dqj
    total_dkmq = dkmqi + dkmqj

    numer = expand(Pi * Pj)
    monoms = collect_monomials(numer)

    raw_terms = []
    for (a_k, b_mu, r_zq, p_q2, s_kmq2), coeff in monoms.items():
        delta_nu1 = total_dq   // 2 - p_q2
        delta_nu2 = total_dkmq // 2 - s_kmq2

        raw_terms.append({
            'coeff':         coeff,
            'explicit_mu':   b_mu,
            'los_rank_r':    r_zq,
            'delta_nu1':     delta_nu1,
            'delta_nu2':     delta_nu2,
            'extra_k_power': a_k,
        })

    return raw_terms


# ════════════════════════════════════════════════════════════════
#  §7  RAW MONOMIAL -> DESCRIPTOR TERM EXPANSION
# ════════════════════════════════════════════════════════════════

def _canonical_term_key(term):
    return (
        term['los_rank_r'],
        term['delta_nu1'],
        term['delta_nu2'],
        term['master_component'],
        term['extra_k_power'],
        term.get('extra_mu_power', 0),
    )


def _combine_descriptor_terms(terms):
    """Merge identical descriptor terms by summing exact rational coefficients."""
    accum = {}
    for term in terms:
        key = _canonical_term_key(term)
        frac = R(term['coeff_rational']['num'], term['coeff_rational']['den'])
        if key not in accum:
            accum[key] = dict(term)
        else:
            prev = accum[key]['coeff_rational']
            frac += R(prev['num'], prev['den'])
        accum[key]['coeff_rational'] = {'num': int(frac.p), 'den': int(frac.q)}

    out = []
    for key, term in sorted(accum.items(), key=lambda kv: kv[0]):
        frac = R(term['coeff_rational']['num'], term['coeff_rational']['den'])
        if frac != 0:
            out.append(term)
    return out


def expand_to_descriptor_terms(raw_terms):
    """Expand raw monomials into final descriptor terms grouped by total mu power.

    For each raw monomial with LOS rank r and explicit mu^b, the master
    integral decomposes into components, each contributing an additional
    implicit mu power.  The total mu power is  b + implicit.

    Returns
    -------
    dict :  total_mu_power -> list of canonicalized term dicts
    """
    terms_by_mu = defaultdict(list)

    for rt in raw_terms:
        r       = rt['los_rank_r']
        b_mu    = rt['explicit_mu']
        coeff   = rt['coeff']
        dnu1    = rt['delta_nu1']
        dnu2    = rt['delta_nu2']
        extra_k = rt['extra_k_power']

        if r > 8:
            raise ValueError(f"LOS rank {r} exceeds maximum 8")

        for comp_name, impl_mu in MASTER_COMPONENTS[r]:
            total_mu = b_mu + impl_mu

            if total_mu % 2 != 0:
                raise ValueError(
                    f"Odd total mu power {total_mu} "
                    f"(explicit={b_mu}, implicit={impl_mu}, r={r})")
            if total_mu > 8:
                raise ValueError(
                    f"Total mu power {total_mu} > 8")

            c = R(coeff)
            term = {
                'coeff_rational':   {'num': int(c.p), 'den': int(c.q)},
                'los_rank_r':       r,
                'delta_nu1':        dnu1,
                'delta_nu2':        dnu2,
                'master_component': comp_name,
                'extra_k_power':    extra_k,
                'extra_mu_power':   0,
            }
            terms_by_mu[total_mu].append(term)

    return {mu_pow: _combine_descriptor_terms(terms)
            for mu_pow, terms in sorted(terms_by_mu.items())}


# ════════════════════════════════════════════════════════════════
#  §8  DESCRIPTOR I/O
# ════════════════════════════════════════════════════════════════

def pair_id(i, j):
    """Packed lower-triangular pair index  (0 <= i <= j < NQ)."""
    return j + i * NQ - i * (i - 1) // 2


def generate_22_descriptors(name_i, name_j, outdir='descriptors'):
    """Generate JSON descriptor files for a 22-side pair channel.

    One file per nonzero mu power.  Returns the list of descriptors.
    """
    raw   = reduce_22_pair_raw(name_i, name_j)
    by_mu = expand_to_descriptor_terms(raw)

    i_idx = Q_NAMES.index(name_i)
    j_idx = Q_NAMES.index(name_j)
    if i_idx > j_idx:
        i_idx, j_idx = j_idx, i_idx
        name_i, name_j = Q_NAMES[i_idx], Q_NAMES[j_idx]

    pid = pair_id(i_idx, j_idx)
    os.makedirs(outdir, exist_ok=True)

    descriptors = []
    for mu_pow in sorted(by_mu):
        terms = by_mu[mu_pow]
        desc = {
            'version':           '1.0',
            'sector':            '22',
            'channel_name':      f'T22_A__{name_i}__{name_j}',
            'channel_id':        pid,
            'mu_power':          mu_pow,
            'k_power_prefactor': 0,
            'radial_rank':       0,
            'symmetry': {
                'type': 'symmetric_pair' if name_i != name_j else 'diagonal_pair',
                'exchange_q_kmq': True,
            },
            'master_family': 'LOS_SHIFTED_J',
            'terms':         terms,
            'metadata': {
                'qi':      name_i,
                'qj':      name_j,
                'pair_id': pid,
                'n_terms': len(terms),
            }
        }
        fname = f'LYA_T22__{name_i}__{name_j}__MU{mu_pow}.json'
        path  = os.path.join(outdir, fname)
        with open(path, 'w') as f:
            json.dump(desc, f, indent=2)
        descriptors.append(desc)

    return descriptors


# ════════════════════════════════════════════════════════════════
#  §9  13-SECTOR CHANNEL DEFINITIONS
# ════════════════════════════════════════════════════════════════
#
# The P13 integrand  int_q K_3(k, q, -q) P_lin(q)  has three kinds
# of contributions (Eq. 3.20):
#
#   A)  Standard F_3, G_3 integrals  ->  import from CLASS-PT
#   B)  New LOS-dependent terms, each multiplied by (1 - (k-hat . q-hat)^2)
#   C)  "Protected" projection terms from Eq. (3.17)
#
# Convention:  each function returns (P, dq, dkmq, denom_k)
# where the channel integrand =  P / (q^dq  kmq^dkmq  k^denom_k)
# and P is a polynomial in {k, mu, zq, q^2, kmq^2}.
#
# The prefactor  (1-(k-hat . q-hat)^2)  is built into P for each channel.

def _angular_prefactor_numer():
    """Numerator of  (1 - (k-hat . q-hat)^2) * 4 k^2 q^2.

    (1 - cos^2 theta_{kq}) = [4k^2 q^2 - (k^2+q^2-kmq^2)^2] / (4k^2 q^2)
    """
    s = k**2 + q**2 - kmq**2
    return expand(4 * k**2 * q**2 - s**2)


def _make_13_channel(coeff, kernel_numer, dq=4, dkmq_pow=2, denom_k=2):
    """Build a 13-sector channel with the universal angular prefactor.

    The returned tuple follows the convention used by reduce_13_channel_raw():
        channel = P / (q^dq * kmq^dkmq_pow * k^denom_k)
    with P polynomial in {k, mu, zq, q^2, kmq^2}.
    """
    ang = _angular_prefactor_numer()
    P = expand(coeff * kernel_numer * ang)
    return (P, dq, dkmq_pow, denom_k)


def t13_DELTAETA():
    r"""-(2/21) [3 (k_parallel-q_parallel)^2 / |k-q|^2 + 5 q_parallel^2 / q^2]."""
    kernel_numer = 3 * zkq**2 * q**2 + 5 * zq**2 * kmq**2
    return _make_13_channel(-R(2, 21), kernel_numer, dq=4, dkmq_pow=2, denom_k=2)


def t13_ETA2():
    r"""(4/7) q_parallel^2 (k_parallel-q_parallel)^2 / (q^2 |k-q|^2)."""
    return _make_13_channel(R(4, 7), zq**2 * zkq**2, dq=4, dkmq_pow=2, denom_k=2)


def t13_KKPAR():
    r"""(20/21) b_(KK)|| channel from Eq. (3.20)."""
    kernel_numer = (
        qdotkmq * zkq * zq
        - zkq**2 * q**2 / 3
        - zq**2 * kmq**2 / 3
        + q**2 * kmq**2 / 9
    )
    return _make_13_channel(R(20, 21), kernel_numer, dq=4, dkmq_pow=2, denom_k=2)


def t13_PI2PAR_MAIN():
    r"""(10/21) (k.q-q^2) (k_parallel-q_parallel)^2 / (|k-q|^2 q^2)."""
    return _make_13_channel(R(10, 21), qdotkmq * zkq**2, dq=4, dkmq_pow=2, denom_k=2)


def t13_DELTAPI2PAR():
    r"""(10/21) (k_parallel-q_parallel)^2 / |k-q|^2."""
    return _make_13_channel(R(10, 21), zkq**2, dq=2, dkmq_pow=2, denom_k=2)


def t13_KPI2PAR_AUX():
    r"""-(10/63) (k_parallel-q_parallel)^2 / |k-q|^2, from the -1/3 b_(KPi[2])|| piece."""
    return _make_13_channel(-R(10, 63), zkq**2, dq=2, dkmq_pow=2, denom_k=2)


def t13_ETAPI2PAR():
    r"""-(10/21) q_parallel^2 (k_parallel-q_parallel)^2 / (q^2 |k-q|^2)."""
    return _make_13_channel(-R(10, 21), zq**2 * zkq**2, dq=4, dkmq_pow=2, denom_k=2)


def t13_KPI2PAR_MAIN():
    r"""(10/21) ((q.k-q^2)/(q |k-q|)) * q_parallel (k_parallel-q_parallel) / (q |k-q|)."""
    return _make_13_channel(R(10, 21), qdotkmq * zq * zkq, dq=4, dkmq_pow=2, denom_k=2)


def t13_PI2PAR_CUBIC():
    r"""(10/21) q_parallel (k_parallel-q_parallel)^3 / (q^2 |k-q|^2)."""
    return _make_13_channel(R(10, 21), zq * zkq**3, dq=4, dkmq_pow=2, denom_k=2)


def t13_PI3PAR():
    r"""(b_{Pi[3]||} + 2 b_{Pi[2]||}) channel from the last line of Eq. (3.20)."""
    kernel_numer = (
        R(13, 21) * qdotkmq * zq * zkq
        - R(5, 9) * mu**2 * (qdotkmq**2 - q**2 * kmq**2 / 3)
    )
    return _make_13_channel(1, kernel_numer, dq=4, dkmq_pow=2, denom_k=2)


def t13_PROJ_B1():
    r"""Protected (2/21) [5 q_parallel(k_parallel-q_parallel)/q^2 + 3 q_parallel(k_parallel-q_parallel)/|k-q|^2]."""
    kernel_numer = 5 * zq * zkq * kmq**2 + 3 * zq * zkq * q**2
    return _make_13_channel(R(2, 21), kernel_numer, dq=4, dkmq_pow=2, denom_k=2)


def t13_PROJ_BETA():
    r"""Protected -(2/7) q_parallel(k_parallel-q_parallel)[(k_parallel-q_parallel)^2 + q_parallel^2]/(q^2 |k-q|^2)."""
    kernel_numer = zq * zkq * (zkq**2 + zq**2)
    return _make_13_channel(-R(2, 7), kernel_numer, dq=4, dkmq_pow=2, denom_k=2)


def t13_F3_B1():
    """Standard F3 channel handled by the pre-existing standard-PT backend."""
    return None


def t13_G3_BETA():
    """Standard G3 channel handled by the pre-existing standard-PT backend."""
    return None


def t13_G2_COMPOSITE():
    r"""(4/21)(5 b_G2 + 2 b_Gamma3) [((k-q).q/(|k-q|q))^2 - 1]."""
    kernel_numer = qdotkmq**2 - q**2 * kmq**2
    return _make_13_channel(R(4, 21), kernel_numer, dq=4, dkmq_pow=2, denom_k=2)


T13_CHANNELS = {
    'T13_B_F3_B1':         t13_F3_B1,
    'T13_B_G3_BETA':       t13_G3_BETA,
    'T13_B_G2_COMPOSITE':  t13_G2_COMPOSITE,
    'T13_B_DELTAETA':      t13_DELTAETA,
    'T13_B_ETA2':          t13_ETA2,
    'T13_B_KKPAR':         t13_KKPAR,
    'T13_B_PI2PAR_MAIN':   t13_PI2PAR_MAIN,
    'T13_B_DELTAPI2PAR':   t13_DELTAPI2PAR,
    'T13_B_KPI2PAR_AUX':   t13_KPI2PAR_AUX,
    'T13_B_ETAPI2PAR':     t13_ETAPI2PAR,
    'T13_B_KPI2PAR_MAIN':  t13_KPI2PAR_MAIN,
    'T13_B_PI2PAR_CUBIC':  t13_PI2PAR_CUBIC,
    'T13_B_PI3PAR':        t13_PI3PAR,
    'T13_B_PROJ_B1':       t13_PROJ_B1,
    'T13_B_PROJ_BETA':     t13_PROJ_BETA,
}
T13_CHANNEL_IDS = {
    name: T13_CHANNEL_MAP[name].channel_id
    for name in T13_CHANNELS
}


# ════════════════════════════════════════════════════════════════
#  §10  13-SECTOR REDUCTION
# ════════════════════════════════════════════════════════════════

def reduce_13_channel_raw(channel_name):
    """Reduce a 13-sector channel to raw monomial form.

    Returns list of term dicts (same schema as reduce_22_pair_raw),
    or None if the channel is a stub (F_3/G_3).
    """
    func = T13_CHANNELS[channel_name]
    result = func()
    if result is None:
        return None

    P, dq, dkmq_pow, denom_k = result
    monoms = collect_monomials(P)

    raw_terms = []
    for (a_k, b_mu, r_zq, p_q2, s_kmq2), coeff in monoms.items():
        net_k = a_k - denom_k

        delta_nu1 = dq       // 2 - p_q2
        delta_nu2 = dkmq_pow // 2 - s_kmq2

        raw_terms.append({
            'coeff':         coeff,
            'explicit_mu':   b_mu,
            'los_rank_r':    r_zq,
            'delta_nu1':     delta_nu1,
            'delta_nu2':     delta_nu2,
            'extra_k_power': net_k,
        })

    return raw_terms


def generate_13_descriptors(channel_name, outdir='descriptors'):
    """Generate JSON descriptors for a 13-sector channel."""
    raw = reduce_13_channel_raw(channel_name)
    if raw is None:
        print(f"  {channel_name}: stub (F3/G3), skipping")
        return []

    by_mu = expand_to_descriptor_terms(raw)

    ch_id = T13_CHANNEL_IDS[channel_name]
    os.makedirs(outdir, exist_ok=True)

    descriptors = []
    for mu_pow in sorted(by_mu):
        terms = by_mu[mu_pow]
        desc = {
            'version':           '1.0',
            'sector':            '13',
            'channel_name':      channel_name,
            'channel_id':        ch_id,
            'mu_power':          mu_pow,
            'k_power_prefactor': 0,
            'radial_rank':       0,
            'symmetry': {
                'type':            'none',
                'exchange_q_kmq':  False,
            },
            'master_family': 'LOS_SHIFTED_J',
            'terms':         terms,
            'metadata': {
                'channel': channel_name,
                'n_terms': len(terms),
            }
        }
        fname = f'LYA_{channel_name}__MU{mu_pow}.json'
        path  = os.path.join(outdir, fname)
        with open(path, 'w') as f:
            json.dump(desc, f, indent=2)
        descriptors.append(desc)

    return descriptors


# ════════════════════════════════════════════════════════════════
#  §11  BIAS CONTRACTION MAP  (22-sector)
# ════════════════════════════════════════════════════════════════

BIAS_CONTRACTION_22 = {
    'DELTA2':   'b2 / 2',
    'G2':       'bG2',
    'F2':       'b1',
    'G2_MU':    'b_eta * f',
    'DELTAETA': 'f * b_delta_eta',
    'ETA2':     'f**2 * b_eta2',
    'RSD_B1':   'b1 * f',
    'RSD_BETA': 'b_eta * f**2',
    'KKPAR':    'b_(KK)_par',
    'PI2PAR':   'b_Pi2_par',
}

BIAS_CONTRACTION_13 = {
    'T13_B_F3_B1':        'b1',
    'T13_B_G3_BETA':      'b_eta * f',
    'T13_B_G2_COMPOSITE': '5 * bG2 + 2 * bGamma3',
    'T13_B_DELTAETA':     'f * b_delta_eta',
    'T13_B_ETA2':         'f**2 * b_eta2',
    'T13_B_KKPAR':        'b_(KK)_par',
    'T13_B_PI2PAR_MAIN':  'b_Pi2_par',
    'T13_B_DELTAPI2PAR':  'b_deltaPi2_par',
    'T13_B_KPI2PAR_AUX':  'b_(KPi2)_par',
    'T13_B_ETAPI2PAR':    'f * b_etaPi2_par',
    'T13_B_KPI2PAR_MAIN': 'b_(KPi2)_par',
    'T13_B_PI2PAR_CUBIC': 'f * b_Pi2_par',
    'T13_B_PI3PAR':       'b_Pi3_par + 2 * b_Pi2_par',
    'T13_B_PROJ_B1':      'f * b1',
    'T13_B_PROJ_BETA':    'f**2 * b_eta',
}


# ════════════════════════════════════════════════════════════════
#  §12  TESTS AND VALIDATION
# ════════════════════════════════════════════════════════════════

def _test_parity(raw_terms, label=""):
    """Verify that  explicit_mu + los_rank_r  is always even."""
    for t in raw_terms:
        s = t['explicit_mu'] + t['los_rank_r']
        assert s % 2 == 0, (
            f"Parity violation in {label}: "
            f"explicit_mu={t['explicit_mu']}, r={t['los_rank_r']}")


def _test_max_los_rank(raw_terms, rmax, label=""):
    for t in raw_terms:
        assert t['los_rank_r'] <= rmax, (
            f"LOS rank {t['los_rank_r']} > {rmax} in {label}")


def run_tests():
    """Self-consistency checks on the reduction engine."""
    print("=" * 60)
    print("Running self-consistency tests")
    print("=" * 60)
    n_pass = 0

    # Test 1: DELTA2 x DELTA2
    raw = reduce_22_pair_raw('DELTA2', 'DELTA2')
    assert len(raw) == 1
    t = raw[0]
    assert t['coeff'] == 1
    assert t['explicit_mu'] == 0
    assert t['los_rank_r'] == 0
    assert t['delta_nu1'] == 0
    assert t['delta_nu2'] == 0
    assert t['extra_k_power'] == 0
    print("  [PASS] DELTA2 x DELTA2 = 1 (trivial)")
    n_pass += 1

    # Test 2: Parity for priority-1 pairs
    priority_pairs = [
        ('DELTAETA', 'DELTAETA'),
        ('DELTAETA', 'ETA2'),
        ('ETA2',     'ETA2'),
        ('KKPAR',    'DELTAETA'),
        ('KKPAR',    'ETA2'),
        ('PI2PAR',   'DELTAETA'),
        ('PI2PAR',   'ETA2'),
        ('KKPAR',    'PI2PAR'),
    ]
    for ni, nj in priority_pairs:
        raw = reduce_22_pair_raw(ni, nj)
        _test_parity(raw, f"{ni} x {nj}")
    print("  [PASS] Parity (explicit_mu + r even) for all priority-1 pairs")
    n_pass += 1

    # Test 3: Max LOS rank bounds
    raw = reduce_22_pair_raw('DELTAETA', 'DELTAETA')
    _test_max_los_rank(raw, 4, "DELTAETA x DELTAETA")
    print("  [PASS] DELTAETA x DELTAETA: max LOS rank <= 4")
    n_pass += 1

    raw = reduce_22_pair_raw('ETA2', 'ETA2')
    _test_max_los_rank(raw, 8, "ETA2 x ETA2")
    print("  [PASS] ETA2 x ETA2: max LOS rank <= 8")
    n_pass += 1

    # Test 4: Symmetry under i <-> j
    raw_ij = reduce_22_pair_raw('DELTAETA', 'ETA2')
    raw_ji = reduce_22_pair_raw('ETA2', 'DELTAETA')
    def _to_set(terms):
        return {(t['explicit_mu'], t['los_rank_r'],
                 t['delta_nu1'], t['delta_nu2'],
                 t['extra_k_power'], t['coeff'])
                for t in terms}
    assert _to_set(raw_ij) == _to_set(raw_ji)
    print("  [PASS] Q_i x Q_j symmetric under i <-> j")
    n_pass += 1

    # Test 5: Even total mu powers
    raw = reduce_22_pair_raw('KKPAR', 'ETA2')
    by_mu = expand_to_descriptor_terms(raw)
    for mu_pow in by_mu:
        assert mu_pow % 2 == 0, f"Odd total mu power {mu_pow}"
        assert 0 <= mu_pow <= 8, f"Total mu power {mu_pow} out of range"
    print("  [PASS] KKPAR x ETA2: all total mu powers even, in [0,8]")
    n_pass += 1

    # Test 6: Counting
    raw = reduce_22_pair_raw('DELTAETA', 'DELTAETA')
    by_mu = expand_to_descriptor_terms(raw)
    total_terms = sum(len(v) for v in by_mu.values())
    print(f"  [INFO] DELTAETA x DELTAETA: {len(raw)} raw -> "
          f"{total_terms} descriptor terms across mu powers {sorted(by_mu.keys())}")
    n_pass += 1

    # Test 7: 13-sector parity
    for ch_name in ['T13_B_DELTAETA', 'T13_B_ETA2', 'T13_B_KKPAR',
                     'T13_B_PI2PAR_MAIN', 'T13_B_DELTAPI2PAR', 'T13_B_KPI2PAR_AUX',
                     'T13_B_ETAPI2PAR', 'T13_B_KPI2PAR_MAIN', 'T13_B_PI2PAR_CUBIC',
                     'T13_B_PI3PAR', 'T13_B_PROJ_B1', 'T13_B_PROJ_BETA',
                     'T13_B_G2_COMPOSITE']:
        raw = reduce_13_channel_raw(ch_name)
        if raw is not None:
            _test_parity(raw, ch_name)
    print("  [PASS] 13-sector parity for all implemented channels")
    n_pass += 1

    # Test 8: 13-sector max LOS rank
    for ch_name in ['T13_B_DELTAETA', 'T13_B_ETA2', 'T13_B_PI3PAR']:
        raw = reduce_13_channel_raw(ch_name)
        if raw is not None:
            _test_max_los_rank(raw, 8, ch_name)
    print("  [PASS] 13-sector max LOS rank <= 8")
    n_pass += 1

    print(f"\nAll {n_pass} tests passed.\n")


# ════════════════════════════════════════════════════════════════
#  §13  SUMMARY REPORT
# ════════════════════════════════════════════════════════════════

def summarize_pair(name_i, name_j):
    """Print a compact summary of the reduction for one 22-pair."""
    raw   = reduce_22_pair_raw(name_i, name_j)
    by_mu = expand_to_descriptor_terms(raw)

    max_r   = max(t['los_rank_r'] for t in raw)
    max_mu  = max(by_mu.keys()) if by_mu else 0
    n_raw   = len(raw)
    n_desc  = sum(len(v) for v in by_mu.values())

    comps = set()
    for terms in by_mu.values():
        for t in terms:
            comps.add(t['master_component'])

    print(f"  {name_i:>10s} x {name_j:<10s}  | "
          f"raw={n_raw:3d}  desc={n_desc:3d}  "
          f"max_r={max_r}  mu_range=[0..{max_mu}]  "
          f"masters={sorted(comps)}")


# ════════════════════════════════════════════════════════════════
#  §14  MAIN DRIVER
# ════════════════════════════════════════════════════════════════

PRIORITY_1_PAIRS = [
    ('DELTAETA', 'DELTAETA'),
    ('DELTAETA', 'ETA2'),
    ('ETA2',     'ETA2'),
    ('KKPAR',    'DELTAETA'),
    ('KKPAR',    'ETA2'),
    ('PI2PAR',   'DELTAETA'),
    ('PI2PAR',   'ETA2'),
    ('KKPAR',    'PI2PAR'),
]

PRIORITY_1_T13 = [
    'T13_B_DELTAETA',
    'T13_B_ETA2',
    'T13_B_KKPAR',
    'T13_B_PI2PAR_MAIN',
    'T13_B_DELTAPI2PAR',
    'T13_B_KPI2PAR_AUX',
    'T13_B_ETAPI2PAR',
    'T13_B_KPI2PAR_MAIN',
    'T13_B_PI2PAR_CUBIC',
    'T13_B_PI3PAR',
    'T13_B_PROJ_B1',
    'T13_B_PROJ_BETA',
    'T13_B_G2_COMPOSITE',
]


def main():
    if '--test' in sys.argv:
        run_tests()
        return

    outdir = 'descriptors'
    print("=" * 70)
    print("LyA one-loop symbolic reduction engine")
    print("=" * 70)

    run_tests()

    # ---- 22-sector ----
    print("=" * 70)
    print("22-SECTOR: Priority-1 pair reductions")
    print("=" * 70)
    print()

    for ni, nj in PRIORITY_1_PAIRS:
        t0 = time.time()
        descs = generate_22_descriptors(ni, nj, outdir=outdir)
        dt = time.time() - t0
        n_terms = sum(len(d['terms']) for d in descs)
        mu_pows = [d['mu_power'] for d in descs]
        print(f"  {ni:>10s} x {nj:<10s}  | "
              f"mu-powers={mu_pows}  total_terms={n_terms:4d}  "
              f"({dt:.2f}s)")
    print()

    # ---- 13-sector ----
    print("=" * 70)
    print("13-SECTOR: Priority-1 channel reductions")
    print("=" * 70)
    print()

    for ch in PRIORITY_1_T13:
        t0 = time.time()
        descs = generate_13_descriptors(ch, outdir=outdir)
        dt = time.time() - t0
        if descs:
            n_terms = sum(len(d['terms']) for d in descs)
            mu_pows = [d['mu_power'] for d in descs]
            print(f"  {ch:<28s}  | "
                  f"mu-powers={mu_pows}  total_terms={n_terms:4d}  "
                  f"({dt:.2f}s)")
    print()

    # ---- Full 22-pair summary ----
    print("=" * 70)
    print(f"FULL 22-PAIR SUMMARY (all {NQ}x({NQ}+1)/2 = "
          f"{NQ*(NQ+1)//2} pairs)")
    print("=" * 70)
    print()
    for i in range(NQ):
        for j in range(i, NQ):
            summarize_pair(Q_NAMES[i], Q_NAMES[j])
    print()

    files = sorted(os.listdir(outdir))
    print(f"Wrote {len(files)} descriptor files to {outdir}/")
    print("First 10:")
    for f in files[:10]:
        print(f"  {f}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")


if __name__ == '__main__':
    main()
