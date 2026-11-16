"""Numerical bounds and the counters consumed by assembly quality gates.

Each tally entry is ``[hits, values_seen, extreme]``. Clipping and flooring happen before downstream computation, and every counter is reset at the start of each pipeline stage.
"""

from __future__ import annotations

import numpy as np

TALLY: dict[str, list[float]] = {}
NAMES = (
    "amp_search_cap",
    "cap_weights_all_capped",
    "empty_control_cells",
    "l2_abs_max",
    "log2_max",
    "log2_overflow",
    "mt_ls_clip",
    "no_eligible_genes",
    "nr_clip",
    "order_sd_min",
    "shape_search_cap",
    "sim_neff_plan",
    "sim_neff_sim",
    "sim_neff_pooling",
    "sim_neff_xsim",
    "sim_sd_min",
    "sim_share_plan",
    "sim_share_sim",
    "sim_share_pooling",
    "sim_share_xsim",
    "sim_z_clip",
    "v_floor_rel",
    "z_abs_max",
)


def reset() -> None:
    TALLY.clear()


def _note(name: str, hits: int, count: int, extreme: float, pick) -> None:
    tally = TALLY.setdefault(name, [0, 0, np.nan])
    tally[0] += int(hits)
    tally[1] += int(count)
    tally[2] = float(extreme if np.isnan(tally[2]) else pick(tally[2], extreme))


def over(values, bound: float, name: str) -> None:
    array = np.abs(np.asarray(values, np.float64))
    with np.errstate(invalid="ignore"):
        extreme = float(np.nanmax(array)) if (~np.isnan(array)).any() else np.nan
    _note(name, int((array > bound).sum()), array.size, extreme, max)


def under(values, bound: float, name: str) -> None:
    array = np.asarray(values, np.float64)
    with np.errstate(invalid="ignore"):
        extreme = float(np.nanmin(array)) if (~np.isnan(array)).any() else np.nan
    _note(name, int((array < bound).sum()), array.size, extreme, min)


def clip_abs(values, bound: float, name: str) -> np.ndarray:
    over(values, bound, name)
    return np.clip(np.asarray(values), -bound, bound)


def floor(values, bound: float, name: str) -> np.ndarray:
    under(values, bound, name)
    return np.maximum(np.asarray(values), bound)


def fired(name: str, hits: int, count: int) -> None:
    _note(name, hits, count, np.nan, max)


def share(lines, weights, operation: str, limits) -> None:
    """Check concentration and effective count of cell-line similarity weights."""
    totals: dict[str, float] = {}
    for line, weight in zip(lines, np.asarray(weights, np.float64)):
        totals[str(line)] = totals.get(str(line), 0.0) + float(weight)
    shares = np.array(list(totals.values())) / sum(totals.values())
    over(shares.max(), limits.sim_share_max[operation], f"sim_share_{operation}")
    under(1.0 / (shares * shares).sum(), limits.sim_neff_min[operation], f"sim_neff_{operation}")


def snapshot() -> dict[str, list[float]]:
    return {key: list(value) for key, value in sorted(TALLY.items())}
