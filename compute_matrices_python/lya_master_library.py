"""Production-oriented master-function support for LyA descriptors.

This module provides a stable numerical implementation of the universal
`J(nu1, nu2)` master integral together with a lightweight dispatcher for the
named LOS master components referenced by the descriptor schema.

For the higher-rank components, the current implementation keeps the same
algebraic placeholder relations that the prototype evaluator used, but moves
them behind a dedicated master-library boundary so the table writer and runtime
descriptor evaluation no longer depend directly on the smoke-test module.

That split lets the repo evolve the analytic `A5...E8` formulas in one place
without touching the descriptor runtime or I/O layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from mpmath import exp, log, loggamma, pi


@lru_cache(maxsize=32768)
def _j_cached(nu1: complex, nu2: complex) -> complex:
    z = (
        loggamma(1.5 - nu1)
        + loggamma(1.5 - nu2)
        + loggamma(nu1 + nu2 - 1.5)
        - loggamma(nu1)
        - loggamma(nu2)
        - loggamma(3.0 - nu1 - nu2)
        - log(8.0 * pi ** 1.5)
    )
    return complex(exp(z))


def J(nu1, nu2):
    return _j_cached(complex(nu1), complex(nu2))


def _js(nu1, nu2, shift1: int = 0, shift2: int = 0):
    """Convenience wrapper for shifted master integrals J[nu1+shift1, nu2+shift2]."""
    return J(nu1 + shift1, nu2 + shift2)


@dataclass
class ShiftedJCache:
    """Per-(nu1, nu2) cache for shifted J evaluations."""

    nu1: complex
    nu2: complex
    Jfunc: callable = J
    cache: dict[tuple[int, int], complex] = field(default_factory=dict)
    total_requests: int = 0
    unique_evaluations: int = 0

    def __post_init__(self) -> None:
        self.nu1 = complex(self.nu1)
        self.nu2 = complex(self.nu2)

    def get(self, shift1: int = 0, shift2: int = 0) -> complex:
        key = (int(shift1), int(shift2))
        self.total_requests += 1
        if key not in self.cache:
            self.cache[key] = self.Jfunc(self.nu1 + key[0], self.nu2 + key[1])
            self.unique_evaluations += 1
        return self.cache[key]

    @property
    def cache_hits(self) -> int:
        return self.total_requests - self.unique_evaluations

    @property
    def hit_fraction(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def stats(self) -> dict[str, float | int]:
        return {
            "total_requests": self.total_requests,
            "unique_evaluations": self.unique_evaluations,
            "cache_hits": self.cache_hits,
            "hit_fraction": self.hit_fraction,
            "unique_shifts": len(self.cache),
        }

    def with_offset(self, shift1: int = 0, shift2: int = 0):
        return _OffsetShiftedJCacheView(self, int(shift1), int(shift2))


class _OffsetShiftedJCacheView:
    """Lightweight shifted view onto a base per-point cache."""

    __slots__ = ("_base_cache", "_shift1", "_shift2")

    def __init__(self, base_cache: ShiftedJCache, shift1: int, shift2: int):
        self._base_cache = base_cache
        self._shift1 = shift1
        self._shift2 = shift2

    def get(self, shift1: int = 0, shift2: int = 0) -> complex:
        return self._base_cache.get(self._shift1 + int(shift1), self._shift2 + int(shift2))


COMPONENT_SHIFTS = {
    "J": [(1.0, 0, 0)],
    "A1": [(0.5, -1, 0), (-0.5, 0, -1), (0.5, 0, 0)],
    "A2": [(-1 / 8, 0, 0), (-1 / 8, 0, -2), (-1 / 8, -2, 0), (1 / 4, 0, -1), (1 / 4, -1, 0), (1 / 4, -1, -1)],
    "B2": [(3.0, 0, 0), (3.0, 0, -2), (3.0, -2, 0), (2.0, 0, -1), (-6.0, -1, 0), (-6.0, -1, -1)],
    "A3": [(-3 / 16, 0, 0), (-3 / 16, -3, 0), (9 / 16, -2, -1), (3 / 16, -2, 0), (-9 / 16, -1, -2), (3 / 8, -1, -1), (3 / 16, -1, 0), (3 / 16, 0, -3), (-9 / 16, 0, -2), (9 / 16, 0, -1)],
    "B3": [(5 / 16, -3, 0), (-15 / 16, -2, -1), (3 / 16, -2, 0), (15 / 16, -1, -2), (-18 / 16, -1, -1), (3 / 16, -1, 0), (-5 / 16, 0, -3), (15 / 16, 0, -2), (-15 / 16, 0, -1), (5 / 16, 0, 0)],
    "A4": [(3 / 128, -4, 0), (-12 / 128, -3, -1), (-12 / 128, -3, 0), (18 / 128, -2, -2), (12 / 128, -2, -1), (18 / 128, -2, 0), (-12 / 128, -1, -3), (12 / 128, -1, -2), (12 / 128, -1, -1), (-12 / 128, -1, 0), (3 / 128, 0, -4), (-12 / 128, 0, -3), (18 / 128, 0, -2), (-12 / 128, 0, -1), (3 / 128, 0, 0)],
    "B4": [(-15 / 64, -4, 0), (60 / 64, -3, -1), (12 / 64, -3, 0), (-90 / 64, -2, -2), (36 / 64, -2, -1), (6 / 64, -2, 0), (60 / 64, -1, -3), (-108 / 64, -1, -2), (36 / 64, -1, -1), (12 / 64, -1, 0), (-15 / 64, 0, -4), (60 / 64, 0, -3), (-90 / 64, 0, -2), (60 / 64, 0, -1), (-15 / 64, 0, 0)],
    "C4": [(35 / 128, -4, 0), (-140 / 128, -3, -1), (20 / 128, -3, 0), (210 / 128, -2, -2), (-180 / 128, -2, -1), (18 / 128, -2, 0), (-140 / 128, -1, -3), (300 / 128, -1, -2), (-180 / 128, -1, -1), (20 / 128, -1, 0), (35 / 128, 0, -4), (-140 / 128, 0, -3), (210 / 128, 0, -2), (-140 / 128, 0, -1), (35 / 128, 0, 0)],
}


def eval_master_component(component: str, nu1, nu2):
    nu1 = complex(nu1)
    nu2 = complex(nu2)
    base = J(nu1, nu2)

    if component == "J":
        return base
    if component == "A1":
        return 0.5 * (
            _js(nu1, nu2, -1, 0)
            - _js(nu1, nu2, 0, -1)
            + _js(nu1, nu2, 0, 0)
        )
    if component == "A2":
        return (-1 / 8) * (
            _js(nu1, nu2, 0, 0)
            + _js(nu1, nu2, 0, -2)
            + _js(nu1, nu2, -2, 0)
            - 2 * _js(nu1, nu2, 0, -1)
            - 2 * _js(nu1, nu2, -1, 0)
            - 2 * _js(nu1, nu2, -1, -1)
        )
    if component == "B2":
        return 3 * (
            _js(nu1, nu2, 0, 0)
            + _js(nu1, nu2, 0, -2)
            + _js(nu1, nu2, -2, 0)
            + (2 / 3) * _js(nu1, nu2, 0, -1)
            - 2 * _js(nu1, nu2, -1, 0)
            - 2 * _js(nu1, nu2, -1, -1)
        )
    if component == "A3":
        return (-3 / 16) * (
            _js(nu1, nu2, 0, 0)
            + _js(nu1, nu2, -3, 0)
            - 3 * _js(nu1, nu2, -2, -1)
            - _js(nu1, nu2, -2, 0)
            + 3 * _js(nu1, nu2, -1, -2)
            - 2 * _js(nu1, nu2, -1, -1)
            - _js(nu1, nu2, -1, 0)
            - _js(nu1, nu2, 0, -3)
            + 3 * _js(nu1, nu2, 0, -2)
            - 3 * _js(nu1, nu2, 0, -1)
        )
    if component == "B3":
        return (1 / 16) * (
            5 * _js(nu1, nu2, -3, 0)
            - 15 * _js(nu1, nu2, -2, -1)
            + 3 * _js(nu1, nu2, -2, 0)
            + 15 * _js(nu1, nu2, -1, -2)
            - 18 * _js(nu1, nu2, -1, -1)
            + 3 * _js(nu1, nu2, -1, 0)
            - 5 * _js(nu1, nu2, 0, -3)
            + 15 * _js(nu1, nu2, 0, -2)
            - 15 * _js(nu1, nu2, 0, -1)
            + 5 * _js(nu1, nu2, 0, 0)
        )
    if component == "A4":
        return (3 / 128) * (
            _js(nu1, nu2, -4, 0)
            - 4 * _js(nu1, nu2, -3, -1)
            - 4 * _js(nu1, nu2, -3, 0)
            + 6 * _js(nu1, nu2, -2, -2)
            + 4 * _js(nu1, nu2, -2, -1)
            + 6 * _js(nu1, nu2, -2, 0)
            - 4 * _js(nu1, nu2, -1, -3)
            + 4 * _js(nu1, nu2, -1, -2)
            + 4 * _js(nu1, nu2, -1, -1)
            - 4 * _js(nu1, nu2, -1, 0)
            + _js(nu1, nu2, 0, -4)
            - 4 * _js(nu1, nu2, 0, -3)
            + 6 * _js(nu1, nu2, 0, -2)
            - 4 * _js(nu1, nu2, 0, -1)
            + _js(nu1, nu2, 0, 0)
        )
    if component == "B4":
        return (-3 / 64) * (
            5 * _js(nu1, nu2, -4, 0)
            - 20 * _js(nu1, nu2, -3, -1)
            - 4 * _js(nu1, nu2, -3, 0)
            + 30 * _js(nu1, nu2, -2, -2)
            - 12 * _js(nu1, nu2, -2, -1)
            - 2 * _js(nu1, nu2, -2, 0)
            - 20 * _js(nu1, nu2, -1, -3)
            + 36 * _js(nu1, nu2, -1, -2)
            - 12 * _js(nu1, nu2, -1, -1)
            - 4 * _js(nu1, nu2, -1, 0)
            + 5 * _js(nu1, nu2, 0, -4)
            - 20 * _js(nu1, nu2, 0, -3)
            + 30 * _js(nu1, nu2, 0, -2)
            - 20 * _js(nu1, nu2, 0, -1)
            + 5 * _js(nu1, nu2, 0, 0)
        )
    if component == "C4":
        return (1 / 128) * (
            35 * _js(nu1, nu2, -4, 0)
            - 140 * _js(nu1, nu2, -3, -1)
            + 20 * _js(nu1, nu2, -3, 0)
            + 210 * _js(nu1, nu2, -2, -2)
            - 180 * _js(nu1, nu2, -2, -1)
            + 18 * _js(nu1, nu2, -2, 0)
            - 140 * _js(nu1, nu2, -1, -3)
            + 300 * _js(nu1, nu2, -1, -2)
            - 180 * _js(nu1, nu2, -1, -1)
            + 20 * _js(nu1, nu2, -1, 0)
            + 35 * _js(nu1, nu2, 0, -4)
            - 140 * _js(nu1, nu2, 0, -3)
            + 210 * _js(nu1, nu2, 0, -2)
            - 140 * _js(nu1, nu2, 0, -1)
            + 35 * _js(nu1, nu2, 0, 0)
        )
    # NOTE:
    # A1..C4 come from `paperog/code_final.tex` Appendix `app:rsdfft`
    # (the original CLASS-PT paper source), while A5..E8 come from
    # `paper1/lyalpha.tex` Appendix `app:master`, which extends the hierarchy
    # to the higher LOS ranks needed for the LyA selection-dependent terms.
    if component == "A5":
        return (15 / 256) * (
            _js(nu1, nu2, -5, 0)
            - 5 * _js(nu1, nu2, -4, -1)
            - 3 * _js(nu1, nu2, -4, 0)
            + 10 * _js(nu1, nu2, -3, -2)
            + 4 * _js(nu1, nu2, -3, -1)
            + 2 * _js(nu1, nu2, -3, 0)
            - 10 * _js(nu1, nu2, -2, -3)
            + 6 * _js(nu1, nu2, -2, -2)
            + 2 * _js(nu1, nu2, -2, -1)
            + 2 * _js(nu1, nu2, -2, 0)
            + 5 * _js(nu1, nu2, -1, -4)
            - 12 * _js(nu1, nu2, -1, -3)
            + 6 * _js(nu1, nu2, -1, -2)
            + 4 * _js(nu1, nu2, -1, -1)
            - 3 * _js(nu1, nu2, -1, 0)
            - _js(nu1, nu2, 0, -5)
            + 5 * _js(nu1, nu2, 0, -4)
            - 10 * _js(nu1, nu2, 0, -3)
            + 10 * _js(nu1, nu2, 0, -2)
            - 5 * _js(nu1, nu2, 0, -1)
            + _js(nu1, nu2, 0, 0)
        )
    if component == "B5":
        return (
            -35 / 128 * _js(nu1, nu2, -5, 0)
            + 175 / 128 * _js(nu1, nu2, -4, -1)
            + 25 / 128 * _js(nu1, nu2, -4, 0)
            - 175 / 64 * _js(nu1, nu2, -3, -2)
            + 25 / 32 * _js(nu1, nu2, -3, -1)
            + 5 / 64 * _js(nu1, nu2, -3, 0)
            + 175 / 64 * _js(nu1, nu2, -2, -3)
            - 225 / 64 * _js(nu1, nu2, -2, -2)
            + 45 / 64 * _js(nu1, nu2, -2, -1)
            + 5 / 64 * _js(nu1, nu2, -2, 0)
            - 175 / 128 * _js(nu1, nu2, -1, -4)
            + 125 / 32 * _js(nu1, nu2, -1, -3)
            - 225 / 64 * _js(nu1, nu2, -1, -2)
            + 25 / 32 * _js(nu1, nu2, -1, -1)
            + 25 / 128 * _js(nu1, nu2, -1, 0)
            + 35 / 128 * _js(nu1, nu2, 0, -5)
            - 175 / 128 * _js(nu1, nu2, 0, -4)
            + 175 / 64 * _js(nu1, nu2, 0, -3)
            - 175 / 64 * _js(nu1, nu2, 0, -2)
            + 175 / 128 * _js(nu1, nu2, 0, -1)
            - 35 / 128 * _js(nu1, nu2, 0, 0)
        )
    if component == "C5":
        return (
            63 / 256 * _js(nu1, nu2, -5, 0)
            - 315 / 256 * _js(nu1, nu2, -4, -1)
            + 35 / 256 * _js(nu1, nu2, -4, 0)
            + 315 / 128 * _js(nu1, nu2, -3, -2)
            - 105 / 64 * _js(nu1, nu2, -3, -1)
            + 15 / 128 * _js(nu1, nu2, -3, 0)
            - 315 / 128 * _js(nu1, nu2, -2, -3)
            + 525 / 128 * _js(nu1, nu2, -2, -2)
            - 225 / 128 * _js(nu1, nu2, -2, -1)
            + 15 / 128 * _js(nu1, nu2, -2, 0)
            + 315 / 256 * _js(nu1, nu2, -1, -4)
            - 245 / 64 * _js(nu1, nu2, -1, -3)
            + 525 / 128 * _js(nu1, nu2, -1, -2)
            - 105 / 64 * _js(nu1, nu2, -1, -1)
            + 35 / 256 * _js(nu1, nu2, -1, 0)
            - 63 / 256 * _js(nu1, nu2, 0, -5)
            + 315 / 256 * _js(nu1, nu2, 0, -4)
            - 315 / 128 * _js(nu1, nu2, 0, -3)
            + 315 / 128 * _js(nu1, nu2, 0, -2)
            - 315 / 256 * _js(nu1, nu2, 0, -1)
            + 63 / 256 * _js(nu1, nu2, 0, 0)
        )
    if component == "A6":
        return (-5 / 1024) * (
            _js(nu1, nu2, -6, 0)
            - 6 * _js(nu1, nu2, -5, -1)
            - 6 * _js(nu1, nu2, -5, 0)
            + 15 * _js(nu1, nu2, -4, -2)
            + 18 * _js(nu1, nu2, -4, -1)
            + 15 * _js(nu1, nu2, -4, 0)
            - 20 * _js(nu1, nu2, -3, -3)
            - 12 * _js(nu1, nu2, -3, -2)
            - 12 * _js(nu1, nu2, -3, -1)
            - 20 * _js(nu1, nu2, -3, 0)
            + 15 * _js(nu1, nu2, -2, -4)
            - 12 * _js(nu1, nu2, -2, -3)
            - 6 * _js(nu1, nu2, -2, -2)
            - 12 * _js(nu1, nu2, -2, -1)
            + 15 * _js(nu1, nu2, -2, 0)
            - 6 * _js(nu1, nu2, -1, -5)
            + 18 * _js(nu1, nu2, -1, -4)
            - 12 * _js(nu1, nu2, -1, -3)
            - 12 * _js(nu1, nu2, -1, -2)
            + 18 * _js(nu1, nu2, -1, -1)
            - 6 * _js(nu1, nu2, -1, 0)
            + _js(nu1, nu2, 0, -6)
            - 6 * _js(nu1, nu2, 0, -5)
            + 15 * _js(nu1, nu2, 0, -4)
            - 20 * _js(nu1, nu2, 0, -3)
            + 15 * _js(nu1, nu2, 0, -2)
            - 6 * _js(nu1, nu2, 0, -1)
            + _js(nu1, nu2, 0, 0)
        )
    if component == "B6":
        return (
            105 / 1024 * _js(nu1, nu2, -6, 0)
            - 315 / 512 * _js(nu1, nu2, -5, -1)
            - 135 / 512 * _js(nu1, nu2, -5, 0)
            + 1575 / 1024 * _js(nu1, nu2, -4, -2)
            + 225 / 512 * _js(nu1, nu2, -4, -1)
            + 135 / 1024 * _js(nu1, nu2, -4, 0)
            - 525 / 256 * _js(nu1, nu2, -3, -3)
            + 225 / 256 * _js(nu1, nu2, -3, -2)
            + 45 / 256 * _js(nu1, nu2, -3, -1)
            + 15 / 256 * _js(nu1, nu2, -3, 0)
            + 1575 / 1024 * _js(nu1, nu2, -2, -4)
            - 675 / 256 * _js(nu1, nu2, -2, -3)
            + 405 / 512 * _js(nu1, nu2, -2, -2)
            + 45 / 256 * _js(nu1, nu2, -2, -1)
            + 135 / 1024 * _js(nu1, nu2, -2, 0)
            - 315 / 512 * _js(nu1, nu2, -1, -5)
            - 675 / 512 * _js(nu1, nu2, -1, -4)
            + 405 / 512 * _js(nu1, nu2, -1, -3)
            + 225 / 512 * _js(nu1, nu2, -1, -2)
            - 135 / 512 * _js(nu1, nu2, -1, -1)
            + 105 / 1024 * _js(nu1, nu2, -1, 0)
            + 105 / 1024 * _js(nu1, nu2, 0, -6)
            - 315 / 512 * _js(nu1, nu2, 0, -5)
            + 225 / 512 * _js(nu1, nu2, 0, -4)
            - 675 / 512 * _js(nu1, nu2, 0, -3)
            + 1575 / 1024 * _js(nu1, nu2, 0, -2)
            - 315 / 512 * _js(nu1, nu2, 0, -1)
            + 105 / 1024 * _js(nu1, nu2, 0, 0)
        )
    if component == "C6":
        return (
            -315 / 1024 * _js(nu1, nu2, -6, 0)
            + 945 / 512 * _js(nu1, nu2, -5, -1)
            + 105 / 512 * _js(nu1, nu2, -5, 0)
            - 4725 / 1024 * _js(nu1, nu2, -4, -2)
            + 525 / 512 * _js(nu1, nu2, -4, -1)
            + 75 / 1024 * _js(nu1, nu2, -4, 0)
            + 1575 / 256 * _js(nu1, nu2, -3, -3)
            - 1575 / 256 * _js(nu1, nu2, -3, -2)
            + 225 / 256 * _js(nu1, nu2, -3, -1)
            + 15 / 256 * _js(nu1, nu2, -3, 0)
            - 4725 / 1024 * _js(nu1, nu2, -2, -4)
            + 2625 / 256 * _js(nu1, nu2, -2, -3)
            - 3375 / 512 * _js(nu1, nu2, -2, -2)
            + 225 / 256 * _js(nu1, nu2, -2, -1)
            + 75 / 1024 * _js(nu1, nu2, -2, 0)
            + 945 / 512 * _js(nu1, nu2, -1, -5)
            - 3675 / 512 * _js(nu1, nu2, -1, -4)
            + 2625 / 256 * _js(nu1, nu2, -1, -3)
            - 1575 / 256 * _js(nu1, nu2, -1, -2)
            + 525 / 512 * _js(nu1, nu2, -1, -1)
            + 105 / 512 * _js(nu1, nu2, -1, 0)
            - 315 / 1024 * _js(nu1, nu2, 0, -6)
            + 945 / 512 * _js(nu1, nu2, 0, -5)
            - 4725 / 1024 * _js(nu1, nu2, 0, -4)
            + 1575 / 256 * _js(nu1, nu2, 0, -3)
            - 4725 / 1024 * _js(nu1, nu2, 0, -2)
            + 945 / 512 * _js(nu1, nu2, 0, -1)
            - 315 / 1024 * _js(nu1, nu2, 0, 0)
        )
    if component == "D6":
        return (
            231 / 1024 * _js(nu1, nu2, -6, 0)
            - 693 / 512 * _js(nu1, nu2, -5, -1)
            + 63 / 512 * _js(nu1, nu2, -5, 0)
            + 3465 / 1024 * _js(nu1, nu2, -4, -2)
            - 945 / 512 * _js(nu1, nu2, -4, -1)
            + 105 / 1024 * _js(nu1, nu2, -4, 0)
            - 1155 / 256 * _js(nu1, nu2, -3, -3)
            + 1575 / 256 * _js(nu1, nu2, -3, -2)
            - 525 / 256 * _js(nu1, nu2, -3, -1)
            + 25 / 256 * _js(nu1, nu2, -3, 0)
            + 3465 / 1024 * _js(nu1, nu2, -2, -4)
            - 2205 / 256 * _js(nu1, nu2, -2, -3)
            + 3675 / 512 * _js(nu1, nu2, -2, -2)
            - 525 / 256 * _js(nu1, nu2, -2, -1)
            + 105 / 1024 * _js(nu1, nu2, -2, 0)
            - 693 / 512 * _js(nu1, nu2, -1, -5)
            + 2835 / 512 * _js(nu1, nu2, -1, -4)
            - 2205 / 256 * _js(nu1, nu2, -1, -3)
            + 1575 / 256 * _js(nu1, nu2, -1, -2)
            - 945 / 512 * _js(nu1, nu2, -1, -1)
            + 63 / 512 * _js(nu1, nu2, -1, 0)
            + 231 / 1024 * _js(nu1, nu2, 0, -6)
            - 693 / 512 * _js(nu1, nu2, 0, -5)
            + 3465 / 1024 * _js(nu1, nu2, 0, -4)
            - 1155 / 256 * _js(nu1, nu2, 0, -3)
            + 3465 / 1024 * _js(nu1, nu2, 0, -2)
            - 693 / 512 * _js(nu1, nu2, 0, -1)
            + 231 / 1024 * _js(nu1, nu2, 0, 0)
        )
    if component == "A7":
        return (-1 / 2048) * (
            35 * _js(nu1, nu2, -7, 0)
            - 7 * _js(nu1, nu2, -6, -1)
            - 5 * _js(nu1, nu2, -6, 0)
            + 21 * _js(nu1, nu2, -5, -2)
            + 18 * _js(nu1, nu2, -5, -1)
            + 9 * _js(nu1, nu2, -5, 0)
            - 35 * _js(nu1, nu2, -4, -3)
            - 15 * _js(nu1, nu2, -4, -2)
            - 9 * _js(nu1, nu2, -4, -1)
            - 5 * _js(nu1, nu2, -4, 0)
            + 35 * _js(nu1, nu2, -3, -4)
            - 20 * _js(nu1, nu2, -3, -3)
            - 6 * _js(nu1, nu2, -3, -2)
            - 4 * _js(nu1, nu2, -3, -1)
            - 5 * _js(nu1, nu2, -3, 0)
            - 21 * _js(nu1, nu2, -2, -5)
            + 45 * _js(nu1, nu2, -2, -4)
            - 18 * _js(nu1, nu2, -2, -3)
            - 6 * _js(nu1, nu2, -2, -2)
            - 9 * _js(nu1, nu2, -2, -1)
            + 9 * _js(nu1, nu2, -2, 0)
            + 7 * _js(nu1, nu2, -1, -6)
            - 30 * _js(nu1, nu2, -1, -5)
            + 45 * _js(nu1, nu2, -1, -4)
            - 20 * _js(nu1, nu2, -1, -3)
            - 15 * _js(nu1, nu2, -1, -2)
            + 18 * _js(nu1, nu2, -1, -1)
            - 5 * _js(nu1, nu2, -1, 0)
            - _js(nu1, nu2, 0, -7)
            + 7 * _js(nu1, nu2, 0, -6)
            - 21 * _js(nu1, nu2, 0, -5)
            + 35 * _js(nu1, nu2, 0, -4)
            - 35 * _js(nu1, nu2, 0, -3)
            + 21 * _js(nu1, nu2, 0, -2)
            - 7 * _js(nu1, nu2, 0, -1)
            + _js(nu1, nu2, 0, 0)
        )
    if component == "B7":
        return (105 / 2048) * (
            3 * _js(nu1, nu2, -7, 0)
            - 21 * _js(nu1, nu2, -6, -1)
            - 7 * _js(nu1, nu2, -6, 0)
            + 63 * _js(nu1, nu2, -5, -2)
            + 14 * _js(nu1, nu2, -5, -1)
            + 3 * _js(nu1, nu2, -5, 0)
            - 105 * _js(nu1, nu2, -4, -3)
            + 35 * _js(nu1, nu2, -4, -2)
            + 5 * _js(nu1, nu2, -4, -1)
            + _js(nu1, nu2, -4, 0)
            + 105 * _js(nu1, nu2, -3, -4)
            - 140 * _js(nu1, nu2, -3, -3)
            + 30 * _js(nu1, nu2, -3, -2)
            + 4 * _js(nu1, nu2, -3, -1)
            + _js(nu1, nu2, -3, 0)
            - 63 * _js(nu1, nu2, -2, -5)
            + 175 * _js(nu1, nu2, -2, -4)
            - 150 * _js(nu1, nu2, -2, -3)
            + 30 * _js(nu1, nu2, -2, -2)
            + 5 * _js(nu1, nu2, -2, -1)
            + 3 * _js(nu1, nu2, -2, 0)
            + 21 * _js(nu1, nu2, -1, -6)
            - 98 * _js(nu1, nu2, -1, -5)
            + 175 * _js(nu1, nu2, -1, -4)
            - 140 * _js(nu1, nu2, -1, -3)
            + 35 * _js(nu1, nu2, -1, -2)
            + 14 * _js(nu1, nu2, -1, -1)
            - 7 * _js(nu1, nu2, -1, 0)
            - 3 * _js(nu1, nu2, 0, -7)
            + 21 * _js(nu1, nu2, 0, -6)
            - 63 * _js(nu1, nu2, 0, -5)
            + 105 * _js(nu1, nu2, 0, -4)
            - 105 * _js(nu1, nu2, 0, -3)
            + 63 * _js(nu1, nu2, 0, -2)
            - 21 * _js(nu1, nu2, 0, -1)
            + 3 * _js(nu1, nu2, 0, 0)
        )
    if component == "C7":
        return (21 / 2048) * (
            33 * _js(nu1, nu2, -7, 0)
            - 231 * _js(nu1, nu2, -6, -1)
            - 21 * _js(nu1, nu2, -6, 0)
            + 693 * _js(nu1, nu2, -5, -2)
            - 126 * _js(nu1, nu2, -5, -1)
            - 7 * _js(nu1, nu2, -5, 0)
            - 1155 * _js(nu1, nu2, -4, -3)
            + 945 * _js(nu1, nu2, -4, -2)
            - 105 * _js(nu1, nu2, -4, -1)
            - 5 * _js(nu1, nu2, -4, 0)
            + 1155 * _js(nu1, nu2, -3, -4)
            - 2100 * _js(nu1, nu2, -3, -3)
            + 1050 * _js(nu1, nu2, -3, -2)
            - 100 * _js(nu1, nu2, -3, -1)
            - 5 * _js(nu1, nu2, -3, 0)
            - 693 * _js(nu1, nu2, -2, -5)
            + 2205 * _js(nu1, nu2, -2, -4)
            - 2450 * _js(nu1, nu2, -2, -3)
            + 1050 * _js(nu1, nu2, -2, -2)
            - 105 * _js(nu1, nu2, -2, -1)
            - 7 * _js(nu1, nu2, -2, 0)
            + 231 * _js(nu1, nu2, -1, -6)
            - 1134 * _js(nu1, nu2, -1, -5)
            + 2205 * _js(nu1, nu2, -1, -4)
            - 2100 * _js(nu1, nu2, -1, -3)
            + 945 * _js(nu1, nu2, -1, -2)
            - 126 * _js(nu1, nu2, -1, -1)
            - 21 * _js(nu1, nu2, -1, 0)
            - 33 * _js(nu1, nu2, 0, -7)
            + 231 * _js(nu1, nu2, 0, -6)
            - 693 * _js(nu1, nu2, 0, -5)
            + 1155 * _js(nu1, nu2, 0, -4)
            - 1155 * _js(nu1, nu2, 0, -3)
            + 693 * _js(nu1, nu2, 0, -2)
            - 231 * _js(nu1, nu2, 0, -1)
            + 33 * _js(nu1, nu2, 0, 0)
        )
    if component == "D7":
        return (1 / 2048) * (
            429 * _js(nu1, nu2, -7, 0)
            - 3003 * _js(nu1, nu2, -6, -1)
            + 231 * _js(nu1, nu2, -6, 0)
            + 9009 * _js(nu1, nu2, -5, -2)
            - 4158 * _js(nu1, nu2, -5, -1)
            + 189 * _js(nu1, nu2, -5, 0)
            - 15015 * _js(nu1, nu2, -4, -3)
            + 17325 * _js(nu1, nu2, -4, -2)
            - 4725 * _js(nu1, nu2, -4, -1)
            + 175 * _js(nu1, nu2, -4, 0)
            + 15015 * _js(nu1, nu2, -3, -4)
            - 32340 * _js(nu1, nu2, -3, -3)
            + 22050 * _js(nu1, nu2, -3, -2)
            - 4900 * _js(nu1, nu2, -3, -1)
            + 175 * _js(nu1, nu2, -3, 0)
            - 9009 * _js(nu1, nu2, -2, -5)
            + 31185 * _js(nu1, nu2, -2, -4)
            - 39690 * _js(nu1, nu2, -2, -3)
            + 22050 * _js(nu1, nu2, -2, -2)
            - 4725 * _js(nu1, nu2, -2, -1)
            + 189 * _js(nu1, nu2, -2, 0)
            + 3003 * _js(nu1, nu2, -1, -6)
            - 15246 * _js(nu1, nu2, -1, -5)
            + 31185 * _js(nu1, nu2, -1, -4)
            - 32340 * _js(nu1, nu2, -1, -3)
            + 17325 * _js(nu1, nu2, -1, -2)
            - 4158 * _js(nu1, nu2, -1, -1)
            + 231 * _js(nu1, nu2, -1, 0)
            - 429 * _js(nu1, nu2, 0, -7)
            + 3003 * _js(nu1, nu2, 0, -6)
            - 9009 * _js(nu1, nu2, 0, -5)
            + 15015 * _js(nu1, nu2, 0, -4)
            - 15015 * _js(nu1, nu2, 0, -3)
            + 9009 * _js(nu1, nu2, 0, -2)
            - 3003 * _js(nu1, nu2, 0, -1)
            + 429 * _js(nu1, nu2, 0, 0)
        )
    if component == "A8":
        return (1 / 32768) * (
            35 * _js(nu1, nu2, -8, 0)
            - 8 * _js(nu1, nu2, -7, -1)
            - 8 * _js(nu1, nu2, -7, 0)
            + 28 * _js(nu1, nu2, -6, -2)
            + 40 * _js(nu1, nu2, -6, -1)
            + 28 * _js(nu1, nu2, -6, 0)
            - 56 * _js(nu1, nu2, -5, -3)
            - 72 * _js(nu1, nu2, -5, -2)
            - 72 * _js(nu1, nu2, -5, -1)
            - 56 * _js(nu1, nu2, -5, 0)
            + 70 * _js(nu1, nu2, -4, -4)
            + 40 * _js(nu1, nu2, -4, -3)
            + 36 * _js(nu1, nu2, -4, -2)
            + 40 * _js(nu1, nu2, -4, -1)
            + 70 * _js(nu1, nu2, -4, 0)
            - 56 * _js(nu1, nu2, -3, -5)
            + 40 * _js(nu1, nu2, -3, -4)
            + 16 * _js(nu1, nu2, -3, -3)
            + 16 * _js(nu1, nu2, -3, -2)
            + 40 * _js(nu1, nu2, -3, -1)
            - 56 * _js(nu1, nu2, -3, 0)
            + 28 * _js(nu1, nu2, -2, -6)
            - 72 * _js(nu1, nu2, -2, -5)
            + 36 * _js(nu1, nu2, -2, -4)
            + 16 * _js(nu1, nu2, -2, -3)
            + 36 * _js(nu1, nu2, -2, -2)
            - 72 * _js(nu1, nu2, -2, -1)
            + 28 * _js(nu1, nu2, -2, 0)
            - 8 * _js(nu1, nu2, -1, -7)
            + 40 * _js(nu1, nu2, -1, -6)
            - 72 * _js(nu1, nu2, -1, -5)
            + 40 * _js(nu1, nu2, -1, -4)
            + 40 * _js(nu1, nu2, -1, -3)
            - 72 * _js(nu1, nu2, -1, -2)
            + 40 * _js(nu1, nu2, -1, -1)
            - 8 * _js(nu1, nu2, -1, 0)
            + _js(nu1, nu2, 0, -8)
            - 8 * _js(nu1, nu2, 0, -7)
            + 28 * _js(nu1, nu2, 0, -6)
            - 56 * _js(nu1, nu2, 0, -5)
            + 70 * _js(nu1, nu2, 0, -4)
            - 56 * _js(nu1, nu2, 0, -3)
            + 28 * _js(nu1, nu2, 0, -2)
            - 8 * _js(nu1, nu2, 0, -1)
            + _js(nu1, nu2, 0, 0)
        )
    if component == "B8":
        return (-1 / 8192) * (
            315 * _js(nu1, nu2, -8, 0)
            - 72 * _js(nu1, nu2, -7, -1)
            - 40 * _js(nu1, nu2, -7, 0)
            + 252 * _js(nu1, nu2, -6, -2)
            + 168 * _js(nu1, nu2, -6, -1)
            + 60 * _js(nu1, nu2, -6, 0)
            - 504 * _js(nu1, nu2, -5, -3)
            - 168 * _js(nu1, nu2, -5, -2)
            - 72 * _js(nu1, nu2, -5, -1)
            - 24 * _js(nu1, nu2, -5, 0)
            + 630 * _js(nu1, nu2, -4, -4)
            - 280 * _js(nu1, nu2, -4, -3)
            - 60 * _js(nu1, nu2, -4, -2)
            - 24 * _js(nu1, nu2, -4, -1)
            - 10 * _js(nu1, nu2, -4, 0)
            - 504 * _js(nu1, nu2, -3, -5)
            + 840 * _js(nu1, nu2, -3, -4)
            - 240 * _js(nu1, nu2, -3, -3)
            - 48 * _js(nu1, nu2, -3, -2)
            - 24 * _js(nu1, nu2, -3, -1)
            - 24 * _js(nu1, nu2, -3, 0)
            + 252 * _js(nu1, nu2, -2, -6)
            - 840 * _js(nu1, nu2, -2, -5)
            + 900 * _js(nu1, nu2, -2, -4)
            - 240 * _js(nu1, nu2, -2, -3)
            - 60 * _js(nu1, nu2, -2, -2)
            - 72 * _js(nu1, nu2, -2, -1)
            + 60 * _js(nu1, nu2, -2, 0)
            - 72 * _js(nu1, nu2, -1, -7)
            + 392 * _js(nu1, nu2, -1, -6)
            - 840 * _js(nu1, nu2, -1, -5)
            + 840 * _js(nu1, nu2, -1, -4)
            - 280 * _js(nu1, nu2, -1, -3)
            - 168 * _js(nu1, nu2, -1, -2)
            + 168 * _js(nu1, nu2, -1, -1)
            - 40 * _js(nu1, nu2, -1, 0)
            + 9 * _js(nu1, nu2, 0, -8)
            - 72 * _js(nu1, nu2, 0, -7)
            + 252 * _js(nu1, nu2, 0, -6)
            - 504 * _js(nu1, nu2, 0, -5)
            + 630 * _js(nu1, nu2, 0, -4)
            - 504 * _js(nu1, nu2, 0, -3)
            + 252 * _js(nu1, nu2, 0, -2)
            - 72 * _js(nu1, nu2, 0, -1)
            + 9 * _js(nu1, nu2, 0, 0)
        )
    if component == "C8":
        return (1 / 16384) * (
            3465 * _js(nu1, nu2, -8, 0)
            - 264 * _js(nu1, nu2, -7, -1)
            - 72 * _js(nu1, nu2, -7, 0)
            + 924 * _js(nu1, nu2, -6, -2)
            + 168 * _js(nu1, nu2, -6, -1)
            + 28 * _js(nu1, nu2, -6, 0)
            - 1848 * _js(nu1, nu2, -5, -3)
            + 504 * _js(nu1, nu2, -5, -2)
            + 56 * _js(nu1, nu2, -5, -1)
            + 8 * _js(nu1, nu2, -5, 0)
            + 2310 * _js(nu1, nu2, -4, -4)
            - 2520 * _js(nu1, nu2, -4, -3)
            + 420 * _js(nu1, nu2, -4, -2)
            + 40 * _js(nu1, nu2, -4, -1)
            + 6 * _js(nu1, nu2, -4, 0)
            - 1848 * _js(nu1, nu2, -3, -5)
            + 4200 * _js(nu1, nu2, -3, -4)
            - 2800 * _js(nu1, nu2, -3, -3)
            + 400 * _js(nu1, nu2, -3, -2)
            + 40 * _js(nu1, nu2, -3, -1)
            + 8 * _js(nu1, nu2, -3, 0)
            + 924 * _js(nu1, nu2, -2, -6)
            - 3528 * _js(nu1, nu2, -2, -5)
            + 4900 * _js(nu1, nu2, -2, -4)
            - 2800 * _js(nu1, nu2, -2, -3)
            + 420 * _js(nu1, nu2, -2, -2)
            + 56 * _js(nu1, nu2, -2, -1)
            + 28 * _js(nu1, nu2, -2, 0)
            - 264 * _js(nu1, nu2, -1, -7)
            + 1512 * _js(nu1, nu2, -1, -6)
            - 3528 * _js(nu1, nu2, -1, -5)
            + 4200 * _js(nu1, nu2, -1, -4)
            - 2520 * _js(nu1, nu2, -1, -3)
            + 504 * _js(nu1, nu2, -1, -2)
            + 168 * _js(nu1, nu2, -1, -1)
            - 72 * _js(nu1, nu2, -1, 0)
            + 33 * _js(nu1, nu2, 0, -8)
            - 264 * _js(nu1, nu2, 0, -7)
            + 924 * _js(nu1, nu2, 0, -6)
            - 1848 * _js(nu1, nu2, 0, -5)
            + 2310 * _js(nu1, nu2, 0, -4)
            - 1848 * _js(nu1, nu2, 0, -3)
            + 924 * _js(nu1, nu2, 0, -2)
            - 264 * _js(nu1, nu2, 0, -1)
            + 33 * _js(nu1, nu2, 0, 0)
        )
    if component == "D8":
        return (-1 / 8192) * (
            3003 * _js(nu1, nu2, -8, 0)
            - 3432 * _js(nu1, nu2, -7, -1)
            - 264 * _js(nu1, nu2, -7, 0)
            + 12012 * _js(nu1, nu2, -6, -2)
            - 1848 * _js(nu1, nu2, -6, -1)
            - 84 * _js(nu1, nu2, -6, 0)
            - 24024 * _js(nu1, nu2, -5, -3)
            + 16632 * _js(nu1, nu2, -5, -2)
            - 1512 * _js(nu1, nu2, -5, -1)
            - 56 * _js(nu1, nu2, -5, 0)
            + 30030 * _js(nu1, nu2, -4, -4)
            - 46200 * _js(nu1, nu2, -4, -3)
            + 18900 * _js(nu1, nu2, -4, -2)
            - 1400 * _js(nu1, nu2, -4, -1)
            - 50 * _js(nu1, nu2, -4, 0)
            - 24024 * _js(nu1, nu2, -3, -5)
            + 64680 * _js(nu1, nu2, -3, -4)
            - 58800 * _js(nu1, nu2, -3, -3)
            + 19600 * _js(nu1, nu2, -3, -2)
            - 1400 * _js(nu1, nu2, -3, -1)
            - 56 * _js(nu1, nu2, -3, 0)
            + 12012 * _js(nu1, nu2, -2, -6)
            - 49896 * _js(nu1, nu2, -2, -5)
            + 79380 * _js(nu1, nu2, -2, -4)
            - 58800 * _js(nu1, nu2, -2, -3)
            + 18900 * _js(nu1, nu2, -2, -2)
            - 1512 * _js(nu1, nu2, -2, -1)
            - 84 * _js(nu1, nu2, -2, 0)
            - 3432 * _js(nu1, nu2, -1, -7)
            + 20328 * _js(nu1, nu2, -1, -6)
            - 49896 * _js(nu1, nu2, -1, -5)
            + 64680 * _js(nu1, nu2, -1, -4)
            - 46200 * _js(nu1, nu2, -1, -3)
            + 16632 * _js(nu1, nu2, -1, -2)
            - 1848 * _js(nu1, nu2, -1, -1)
            - 264 * _js(nu1, nu2, -1, 0)
            + 429 * _js(nu1, nu2, 0, -8)
            - 3432 * _js(nu1, nu2, 0, -7)
            + 12012 * _js(nu1, nu2, 0, -6)
            - 24024 * _js(nu1, nu2, 0, -5)
            + 30030 * _js(nu1, nu2, 0, -4)
            - 24024 * _js(nu1, nu2, 0, -3)
            + 12012 * _js(nu1, nu2, 0, -2)
            - 3432 * _js(nu1, nu2, 0, -1)
            + 429 * _js(nu1, nu2, 0, 0)
        )
    if component == "E8":
        return (1 / 32768) * (
            6435 * _js(nu1, nu2, -8, 0)
            - 51480 * _js(nu1, nu2, -7, -1)
            + 3432 * _js(nu1, nu2, -7, 0)
            + 180180 * _js(nu1, nu2, -6, -2)
            - 72072 * _js(nu1, nu2, -6, -1)
            + 2772 * _js(nu1, nu2, -6, 0)
            - 360360 * _js(nu1, nu2, -5, -3)
            + 360360 * _js(nu1, nu2, -5, -2)
            - 83160 * _js(nu1, nu2, -5, -1)
            + 2520 * _js(nu1, nu2, -5, 0)
            + 450450 * _js(nu1, nu2, -4, -4)
            - 840840 * _js(nu1, nu2, -4, -3)
            + 485100 * _js(nu1, nu2, -4, -2)
            - 88200 * _js(nu1, nu2, -4, -1)
            + 2450 * _js(nu1, nu2, -4, 0)
            - 360360 * _js(nu1, nu2, -3, -5)
            + 1081080 * _js(nu1, nu2, -3, -4)
            - 1164240 * _js(nu1, nu2, -3, -3)
            + 529200 * _js(nu1, nu2, -3, -2)
            - 88200 * _js(nu1, nu2, -3, -1)
            + 2520 * _js(nu1, nu2, -3, 0)
            + 180180 * _js(nu1, nu2, -2, -6)
            - 792792 * _js(nu1, nu2, -2, -5)
            + 1372140 * _js(nu1, nu2, -2, -4)
            - 1164240 * _js(nu1, nu2, -2, -3)
            + 485100 * _js(nu1, nu2, -2, -2)
            - 83160 * _js(nu1, nu2, -2, -1)
            + 2772 * _js(nu1, nu2, -2, 0)
            - 51480 * _js(nu1, nu2, -1, -7)
            + 312312 * _js(nu1, nu2, -1, -6)
            - 792792 * _js(nu1, nu2, -1, -5)
            + 1081080 * _js(nu1, nu2, -1, -4)
            - 840840 * _js(nu1, nu2, -1, -3)
            + 360360 * _js(nu1, nu2, -1, -2)
            - 72072 * _js(nu1, nu2, -1, -1)
            + 3432 * _js(nu1, nu2, -1, 0)
            + 6435 * _js(nu1, nu2, 0, -8)
            - 51480 * _js(nu1, nu2, 0, -7)
            + 180180 * _js(nu1, nu2, 0, -6)
            - 360360 * _js(nu1, nu2, 0, -5)
            + 450450 * _js(nu1, nu2, 0, -4)
            - 360360 * _js(nu1, nu2, 0, -3)
            + 180180 * _js(nu1, nu2, 0, -2)
            - 51480 * _js(nu1, nu2, 0, -1)
            + 6435 * _js(nu1, nu2, 0, 0)
        )

    raise ValueError(f"Unsupported master component {component!r}")


def eval_master_component_cached(component: str, jcache: ShiftedJCache):
    """Evaluate a master component using a per-point shifted-J cache."""

    js = jcache.get

    if component == "J":
        return js(0, 0)
    if component == "A1":
        return 0.5 * (js(-1, 0) - js(0, -1) + js(0, 0))
    if component == "A2":
        return (-1 / 8) * (js(0, 0) + js(0, -2) + js(-2, 0) - 2 * js(0, -1) - 2 * js(-1, 0) - 2 * js(-1, -1))
    if component == "B2":
        return 3 * (js(0, 0) + js(0, -2) + js(-2, 0) + (2 / 3) * js(0, -1) - 2 * js(-1, 0) - 2 * js(-1, -1))
    if component == "A3":
        return (-3 / 16) * (
            js(0, 0) + js(-3, 0) - 3 * js(-2, -1) - js(-2, 0) + 3 * js(-1, -2) - 2 * js(-1, -1)
            - js(-1, 0) - js(0, -3) + 3 * js(0, -2) - 3 * js(0, -1)
        )
    if component == "B3":
        return (1 / 16) * (
            5 * js(-3, 0) - 15 * js(-2, -1) + 3 * js(-2, 0) + 15 * js(-1, -2) - 18 * js(-1, -1)
            + 3 * js(-1, 0) - 5 * js(0, -3) + 15 * js(0, -2) - 15 * js(0, -1) + 5 * js(0, 0)
        )
    if component == "A4":
        return (3 / 128) * (
            js(-4, 0) - 4 * js(-3, -1) - 4 * js(-3, 0) + 6 * js(-2, -2) + 4 * js(-2, -1) + 6 * js(-2, 0)
            - 4 * js(-1, -3) + 4 * js(-1, -2) + 4 * js(-1, -1) - 4 * js(-1, 0) + js(0, -4) - 4 * js(0, -3)
            + 6 * js(0, -2) - 4 * js(0, -1) + js(0, 0)
        )
    if component == "B4":
        return (-3 / 64) * (
            5 * js(-4, 0) - 20 * js(-3, -1) - 4 * js(-3, 0) + 30 * js(-2, -2) - 12 * js(-2, -1) - 2 * js(-2, 0)
            - 20 * js(-1, -3) + 36 * js(-1, -2) - 12 * js(-1, -1) - 4 * js(-1, 0) + 5 * js(0, -4)
            - 20 * js(0, -3) + 30 * js(0, -2) - 20 * js(0, -1) + 5 * js(0, 0)
        )
    if component == "C4":
        return (1 / 128) * (
            35 * js(-4, 0) - 140 * js(-3, -1) + 20 * js(-3, 0) + 210 * js(-2, -2) - 180 * js(-2, -1) + 18 * js(-2, 0)
            - 140 * js(-1, -3) + 300 * js(-1, -2) - 180 * js(-1, -1) + 20 * js(-1, 0) + 35 * js(0, -4)
            - 140 * js(0, -3) + 210 * js(0, -2) - 140 * js(0, -1) + 35 * js(0, 0)
        )

    return eval_master_component(component, jcache.nu1, jcache.nu2)
