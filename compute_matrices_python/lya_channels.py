"""LyA basis-channel definitions for future CLASS-PT matrix generation.

This module is intentionally metadata-only for the first milestone. It freezes
stable names and ids so that descriptor generation, numerical matrix
generation, and backend loading can all share the same channel vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class T13Channel:
    channel_id: int
    name: str
    mu_powers: tuple[int, ...]
    notes: str


T13_CHANNELS: tuple[T13Channel, ...] = (
    T13Channel(0, "T13_B_F3_B1", (0,), "selection-free b1 F3 contribution"),
    T13Channel(1, "T13_B_G3_BETA", (2,), "selection-free -f b_eta mu^2 G3 contribution"),
    T13Channel(2, "T13_B_G2_COMPOSITE", (0, 2, 4), "placeholder split for future bG2 contraction"),
    T13Channel(3, "T13_B_GAMMA3_ONLY", (0, 2, 4), "placeholder split for future bGamma3 contraction"),
    T13Channel(4, "T13_B_DELTAETA", (0, 2), "direct LOS selection contribution"),
    T13Channel(5, "T13_B_ETA2", (0, 2, 4), "direct LOS quadratic selection contribution"),
    T13Channel(6, "T13_B_KKPAR", (0, 2, 4), "new LOS tensor structure"),
    T13Channel(7, "T13_B_PI2PAR_MAIN", (0, 2, 4), "main cubic Pi2_parallel structure"),
    T13Channel(8, "T13_B_DELTAPI2PAR", (0, 2, 4), "delta times Pi2_parallel structure"),
    T13Channel(9, "T13_B_ETAPI2PAR", (0, 2, 4, 6), "eta times Pi2_parallel structure"),
    T13Channel(10, "T13_B_KPI2PAR_MAIN", (0, 2, 4, 6), "main K Pi2_parallel structure"),
    T13Channel(11, "T13_B_KPI2PAR_AUX", (0, 2, 4, 6), "auxiliary K Pi2_parallel structure"),
    T13Channel(12, "T13_B_PI2PAR_CUBIC", (0, 2, 4, 6), "cubic Pi2_parallel structure"),
    T13Channel(13, "T13_B_PI3PAR", (0, 2, 4, 6), "Pi3_parallel structure"),
    T13Channel(14, "T13_B_PROJ_B1", (0, 2), "protected observer-frame projection term for b1"),
    T13Channel(15, "T13_B_PROJ_BETA", (0, 2, 4), "protected observer-frame projection term for b_eta"),
)


INITIAL_T13_SUBSET: tuple[str, ...] = (
    "T13_B_F3_B1",
    "T13_B_G3_BETA",
    "T13_B_G2_COMPOSITE",
    "T13_B_DELTAETA",
    "T13_B_ETA2",
    "T13_B_KKPAR",
    "T13_B_PI2PAR_MAIN",
    "T13_B_DELTAPI2PAR",
    "T13_B_ETAPI2PAR",
    "T13_B_KPI2PAR_MAIN",
    "T13_B_KPI2PAR_AUX",
    "T13_B_PI2PAR_CUBIC",
    "T13_B_PI3PAR",
    "T13_B_PROJ_B1",
    "T13_B_PROJ_BETA",
)


T13_CHANNEL_MAP = {channel.name: channel for channel in T13_CHANNELS}
