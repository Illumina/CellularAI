"""Per-perturbation placement targets and synthesis jobs.

Each eligible gene gets a target mean per-cell CPM from a detection-limit magnitude rule (model.placement: `rule`
in rule_band, `frozen` elsewhere and x boost in boost_band, all x damp in damp_band), with planned genes forced to
the plan sign at >= max(min_lfc, the frozen rule's magnitude at plan_floor_z).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from .. import guards
from ..task import NCELL
from . import finite


@dataclass(frozen=True)
class Targets:
    """Every per-context placement input; (genes, perts) unless noted."""

    l2: np.ndarray  # pooled-evidence l2
    z: np.ndarray  # pooled-evidence z, finite
    az: np.ndarray  # |z|
    zsim: np.ndarray  # similarity-weighted training-cell-line z
    lq: np.ndarray  # equal-library log2 fold change of the amplitude design's totals, clipped to +-30
    cpm: np.ndarray  # (G,) control mean CPM (the scorer's)
    elig: np.ndarray  # (G,) eligible genes
    log2_cg: np.ndarray  # (G,)
    depth: float  # median control depth
    amp: float  # the amplitude design's factor
    lib: np.ndarray  # (n_cells,) target library sizes
    tidx: np.ndarray  # (P,) own-target gene index, -1 off the axis


def pert_rng(pert: str, seed: int | None = None):
    """A perturbation's synthesis generator: from its name alone, or from its name and a task seed."""
    key = zlib.crc32(pert.encode())
    return np.random.default_rng(key if seed is None else [key, seed])


def library_profile(ctrl_libs, s_tot: int, spread: float, n_cells: int = NCELL) -> np.ndarray:
    """n_cells target library sizes: control library quantiles (k + 0.5)/n_cells, spread ^spread about the median,
    sum s_tot."""
    qs = (np.arange(n_cells) + 0.5) / n_cells
    libs = np.asarray(ctrl_libs, dtype=np.float64)
    if (libs > 0).any():  # empty control cells would give zero target libraries
        guards.fired("empty_control_cells", int((libs <= 0).sum()), libs.size)
        libs = libs[libs > 0]
    LT = np.quantile(libs, qs)
    LT = LT * ((s_tot / n_cells) / LT.mean())
    LT = np.median(LT) * np.power(LT / np.median(LT), spread)
    LT = LT * (s_tot / LT.sum())
    LT[-1] += s_tot - int(round(LT.sum()))
    return LT


def _magnitude(pr, z, cpm, depth, l2=None):
    mu = np.maximum(np.asarray(cpm, np.float64) * depth / 1e6, 1e-3)
    z2 = np.asarray(z, np.float64) ** 2
    m = pr.amp * (pr.h0 + (1 - pr.h0) * z2 / (z2 + pr.kz)) * np.sqrt(1.0 / mu + pr.phi)
    m = np.minimum(m, pr.cap)
    if l2 is not None:
        m = np.maximum(m, pr.cl2 * np.abs(l2))
    return m


def _comp_offset(t, k) -> float:
    """c_k: median equal-library lfc + log2_cg over eligible genes without pooled-evidence evidence (l2 == 0)."""
    m = (t.l2[:, k] == 0) & t.elig
    return float(np.median((t.lq[:, k] + t.log2_cg)[m])) if m.any() else 0.0


def _rule_target(pr, t, k, evidence_z):
    """sign(amp l2 + c_k) x magnitude; NaN (no placement) without pooled-evidence evidence behind the sign."""
    s1 = np.sign(t.lq[:, k])
    s1[s1 == 0] = 1.0
    ck = _comp_offset(t, k)
    s = np.sign(t.amp * t.l2[:, k] + ck + 0.0)
    s = np.where(s == 0, s1, s)
    x = s * _magnitude(pr, t.z[:, k], t.cpm, t.depth, t.l2[:, k])
    z = np.abs(np.where(np.isfinite(t.z[:, k]), t.z[:, k], 0.0))
    if not (z[t.elig] >= evidence_z).any():
        return np.full_like(x, np.nan)
    if ck == 0.0:
        x = np.where(t.l2[:, k] == 0, np.nan, x)
    return x


def _in(z, band):
    return (z >= band[0]) & (z < band[1])


def dn_target(t, k, m):
    p = m.placement
    x = _rule_target(p.rule, t, k, p.evidence_z)
    z = np.abs(finite(t.z[:, k]))
    dn = _rule_target(p.frozen, t, k, p.evidence_z)
    res = np.where(_in(z, p.rule_band) & np.isfinite(x), x, dn)
    res = np.where(_in(z, p.boost_band) & np.isfinite(dn), dn * p.boost, res)
    return np.asarray(res, np.float64) * np.where(_in(z, p.damp_band), p.damp, 1.0)


def plan_sign(ls, idx, sg, t, m):
    ls = np.asarray(ls, np.float64).copy()
    lo = np.maximum(
        np.full(idx.size, m.calls.min_lfc),
        _magnitude(m.placement.frozen, np.full(idx.size, m.placement.plan_floor_z), np.asarray(t.cpm)[idx], t.depth),
    )
    ls[idx] = sg * np.maximum(sg * ls[idx], lo)
    return ls


def q_order(az, s, zs, p):
    """Ordering key: pooled-evidence |z| and the signed similarity-weighted training-cell-line z."""
    return np.log1p(az) / p.order_az + np.clip(s * zs, *p.order_clip) / p.order_zsim


def job(k, q, c, t: Targets, T, m, seed: int | None = None) -> dict:
    """The synthesis job of perturbation k (name q): plan entry c, totals T (P, G)."""
    idx = np.asarray(c["idx"], dtype=np.int64)
    plan_sg = np.sign(np.asarray(c["sign"], dtype=np.float64))
    pri = c["priority"]
    if len(c["idx"]) > 1:
        zi = np.nan_to_num(np.asarray(t.z, dtype=np.float64)[idx, k], nan=0.0)
        zi = q_order(np.abs(zi), plan_sg, t.zsim[idx, k], m.placement)
        o_ = np.lexsort((np.arange(zi.size), -zi))
        pri = np.empty(zi.size)
        pri[o_] = 1.0 - np.arange(zi.size) / zi.size
    sg = plan_sg.copy()
    sg[sg == 0] = 1.0
    ls = plan_sign(dn_target(t, k, m), idx, sg, t, m)
    mt = t.cpm * np.power(2.0, guards.clip_abs(np.nan_to_num(ls, nan=0.0), m.guards.mt_ls_clip, "mt_ls_clip"))
    mt[~np.isfinite(ls) | ~t.elig | (T[k] == 0)] = -1.0
    st_ = np.asarray(c["nudge_strength"], dtype=np.float64)
    strength = c["nudge_strength"]
    rest_ = np.flatnonzero(st_ > 0)
    if rest_.size > 1:
        # Unplanned genes' nudge ranks follow the q-order key on their placement sign.
        lsn_ = np.sign(np.nan_to_num(ls, nan=0.0))
        s_ = np.where((mt > 0) & (lsn_ != 0), lsn_, np.sign(np.asarray(c["nudge_sign"], dtype=np.float64)))
        s_[idx] = plan_sg
        key_ = q_order(t.az[:, k], s_, t.zsim[:, k], m.placement)
        o_ = rest_[np.lexsort((-st_[rest_], -key_[rest_]))]
        strength = np.zeros_like(st_)
        strength[o_] = 1.0 - (np.arange(o_.size) + 0.5) / o_.size
    if t.tidx[k] >= 0:
        mt[t.tidx[k]] = -1.0
    return dict(
        T=T[k],
        plan_idx=c["idx"],
        plan_sign=c["sign"],
        plan_priority=pri,
        nudge_sign=c["nudge_sign"],
        nudge_strength=strength,
        rng=pert_rng(q, seed),
        Lt=t.lib,
        mt=mt,
    )
