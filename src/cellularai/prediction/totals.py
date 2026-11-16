"""The per-perturbation aggregate-count design and the credit it is solved against.

Designs are log2 factors L (genes x perturbations) on the control pool's gene fractions. Their profile spread
(squared log1p(TS * fraction) deviation from the per-gene panel mean, own targets masked) scales with the panel's
credit, the perturbations x the mean delete-one jackknife moment of n_cells random control cells (amp_share of it sets
the amplitude). The design's margin (its spread target / credit) is solved so the scorer's spread_B of the integer totals it yields
is the scorer's expected credit sum_p (1/G_p) E min(C^_p, C_p): C^_p the moment of our synthesized cells for p,
C_p sampled by the control draws (re-solved to a fixed point, since C^_p follows the synthesis).
"""

from __future__ import annotations

import numpy as np

from ..evaluation.scoring import TS, jackknife_moment
from ..task import NCELL
from . import exp2, finite

SOLVE_RTOL, SOLVE_STEPS = 1e-4, 12


def control_moments(ctrl, t, n_cells: int = NCELL) -> np.ndarray:
    """(credit_draws,) jackknife moments of seeded n_cells-control-cell draws: samples of a perturbation's truth C_p."""
    rng = np.random.default_rng(t.credit_seed)
    n = ctrl.shape[0]
    return np.array(
        [jackknife_moment(ctrl[rng.choice(n, n_cells, replace=n < n_cells)]) for _ in range(t.credit_draws)]
    )


def expected_credit(c_hat, draws, w) -> float:
    """The scorer's credit sum_p w_p min(C^_p, C_p) in expectation over C_p ~ draws; w_p = 1/G_p."""
    return float(np.mean([(w * np.minimum(c_hat, d)).sum() for d in draws]))


def own_mask(shape, tidx) -> np.ndarray:
    """False at each perturbation's own target gene."""
    m = np.ones(shape, dtype=bool)
    for k, t in enumerate(tidx):
        if t >= 0:
            m[t, k] = False
    return m


def profile_spread(fr, mask) -> float:
    """sum of squared log1p(TS fr) deviations from the per-gene panel mean; fr (genes, perts) fractions."""
    Bm = np.log1p(TS * fr)
    mu = np.where(mask, Bm, 0).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)
    return float((np.where(mask, Bm - mu, 0.0) ** 2).sum())


def make_spread(fc, mask, log2_max):
    def spread(L):
        fr = fc[:, None] * exp2(L, log2_max)
        return profile_spread(fr / fr.sum(axis=0, keepdims=True), mask)

    return spread


def totals_spread(T, mask, w) -> float:
    """The scorer's spread_B of integer totals T (perts, genes): w_p-weighted squared log1p profile deviations from
    the w-weighted per-gene panel mean, own targets masked; w_p = 1/G_p."""
    T = np.asarray(T, np.float64)
    B = np.log1p(TS * T / T.sum(axis=1, keepdims=True))
    wm = np.where(mask.T, np.asarray(w, np.float64)[:, None], 0.0)
    ws = wm.sum(axis=0)
    mu = np.where(ws > 0, (wm * B).sum(axis=0) / np.maximum(ws, 1e-300), 0.0)
    return float((wm * (B - mu) ** 2).sum())


def solve_margin(realized, target: float, start: float) -> tuple[float, bool]:
    """(m, reached): the margin m >= 0 with realized(m) = target by secant steps (realized spread is near-linear in
    m), else by bisection on [0, start 2^SOLVE_STEPS]; an unreachable target clamps m to that bound, reached False;
    reached is True only if |realized(m) / target - 1| <= SOLVE_RTOL."""
    m0, r0 = start, realized(start)
    m1 = m0 * target / r0 if r0 > 0 else -1.0
    for _ in range(SOLVE_STEPS):
        if not m1 > 0:
            break
        r1 = realized(m1)
        if abs(r1 / target - 1) <= SOLVE_RTOL:
            return m1, True
        if r1 == r0:
            break
        m0, r0, m1 = m1, r1, m1 + (target - r1) * (m1 - m0) / (r1 - r0)
    lo, hi = 0.0, start
    if realized(lo) >= target:
        return lo, False
    while (rhi := realized(hi)) < target:
        if hi >= start * 2**SOLVE_STEPS:
            return hi, False
        lo, hi = hi, 2 * hi
    for _ in range(4 * SOLVE_STEPS):
        mid = 0.5 * (lo + hi)
        r = realized(mid)
        if abs(r / target - 1) <= SOLVE_RTOL:
            return mid, True
        lo, hi, rhi = (mid, hi, rhi) if r < target else (lo, mid, r)
    return hi, abs(rhi / target - 1) <= SOLVE_RTOL


def bisect_amp(spread, design, target) -> tuple[float, bool]:
    """(a, capped): the bisected amplitude with spread(design(a)) >= target; capped (a = 64) when even 64 falls
    short."""
    lo, hi = 0.0, 1.0
    while (s := spread(design(hi))) < target and hi < 64:
        hi *= 2
    capped = s < target
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if spread(design(mid)) < target else (lo, mid)
    return hi, capped


def _base(l2, gamma):
    return np.sign(l2) * np.power(np.abs(l2), gamma)


def _zshape(z, kz):
    return z * z / (z * z + kz)


def control_pcs(ctrl, k: int, n_cells: int, seed: int):
    """(k, G) top gene-space PCs of n_cells seeded control cells' centred log1p(TS x / lib), and sv**2."""
    X = np.asarray(ctrl)
    idx = np.sort(np.random.default_rng(seed).choice(X.shape[0], min(n_cells, X.shape[0]), replace=False))
    Xs = np.asarray(X[idx], dtype=np.float64)
    C = np.log1p(TS * Xs / np.maximum(Xs.sum(1, keepdims=True), 1e-12))
    C -= C.mean(0)
    _, sv, Vt = np.linalg.svd(C, full_matrices=False)
    return Vt[:k], (sv**2)[:k]


def strength_gate(z, L0, mask, top, gate_z) -> np.ndarray:
    """(P,) 1.0 for the perturbations in the top `top` share by n(|z| >= gate_z) (ties by the shape's deviation norm)."""
    P = z.shape[1]
    lb = np.where(mask, L0, 0.0).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)
    nrm = np.sqrt((np.where(mask, L0 - lb, 0.0) ** 2).sum(0))
    s = (np.abs(np.nan_to_num(z)) >= gate_z).sum(0) + 1e-3 * nrm / max(float(nrm.max()), 1e-12)
    r = np.argsort(np.argsort(s, kind="stable"), kind="stable") / max(P - 1, 1)
    return (r >= 1.0 - top).astype(np.float64)


def cp_design(L, fc, mask, Vb, lam, gate, spread, target):
    """D' = D + lam g_p V'V D on the masked deviations, rescaled to the spread target; own targets keep L."""
    fr = fc[:, None] * exp2(L, None)
    nrm = np.log2(fr.sum(axis=0, keepdims=True))
    Bm = np.log1p(TS * fr / fr.sum(axis=0, keepdims=True))
    mu = np.where(mask, Bm, 0).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)
    D = Bm - mu
    Dn = np.where(mask, D + lam * (Vb.T @ (Vb @ np.where(mask, D, 0.0))) * gate[None, :], D)
    pos = fc > 0

    def to_L(c):
        f = np.expm1(np.maximum(mu + c * Dn, 0.0)) / TS
        with np.errstate(divide="ignore"):
            Ln = np.log2(np.maximum(f, 1e-300) / np.where(pos, fc, 1.0)[:, None]) + nrm
        return np.where(pos[:, None] & mask, np.maximum(Ln, -60.0), L)

    lo, hi = 0.0, 1.0
    while (s := spread(to_L(hi))) < target and hi < 256:
        hi *= 2
    capped = s < target
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if spread(to_L(mid)) < target else (lo, mid)
    return to_L(hi), capped


def shaped_design(l2_tot, z_tot, z, profile, fc, mask, pcs, spread, target, t):
    """(L, (amp capped, CP capped, PW capped)): the totals shape, averaged with the same shape on the profile effects,
    amplitude-matched, then the control-program smoothing (CP) and program weighting (PW) at the same spread target;
    pcs = control_pcs(.., pw_k)."""
    zz = finite(z_tot)
    L0 = _base(l2_tot, t.shape_gamma) * _zshape(zz, t.shape_kz)
    za = finite(z)
    L0 = (1 - t.mix) * L0 + t.mix * _base(profile, t.shape_gamma) * _zshape(za, t.shape_kz)
    lbar = np.where(mask, L0, 0.0).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)

    def design(a):
        return lbar + a * (L0 - lbar)

    a, c_amp = bisect_amp(spread, design, target)
    Vt, ev = pcs
    g = strength_gate(zz, L0, mask, t.cp_top, t.gate_z)
    L, c_cp = cp_design(design(a), fc, mask, Vt[t.cp_j0 : t.cp_k], t.cp_lam, g, spread, target)
    w = (ev / ev[0]) ** t.pw_gamma
    g = strength_gate(zz, L0, mask, t.pw_top, t.gate_z)
    L, c_pw = cp_design(L, fc, mask, np.sqrt(w)[:, None] * Vt, t.pw_lam, g, spread, target)
    return L, (c_amp, c_cp, c_pw)
