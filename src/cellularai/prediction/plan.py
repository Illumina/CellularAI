"""The call plan: which genes each perturbation calls, with which sign and priority, and how many.

Orderings: training-cell-line lines are weighted by the similarity of their control profile to
this context's controls; the plan z is the |Stouffer| of the training-cell-line z's at half-uniform / half-similarity weights
over every line (zero-filled), the similarity z the capped-weight Stouffer over the measuring lines.
Sizes: the training-data-breadth curve at the task median_degs,
PS-reassigned and bottom-shrunk by a power score, then raised toward the curve evaluated on the
similarity-weighted breadth (x_sim, mode 'addc'). Membership: the top-scored eligible genes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .. import guards
from . import finite
from .totals import own_mask


# ------------------------------------------------------------------ training-cell-line similarity and orderings
def _on_axis(genes, line_genes, values):
    pos = {g: i for i, g in enumerate(line_genes)}
    idx = np.array([pos.get(g, -1) for g in genes])
    out = np.full(len(genes), np.nan)
    out[idx >= 0] = np.asarray(values)[idx[idx >= 0]]
    return out


def similarity_weights(genes, target_cpm, reference_ctrl, lines, tau, ntop, sd_min):
    """Softmax of the z-scored Pearson r of log1p control CPM on the ntop most training-cell-line-variable genes, the sd of
    r floored at sd_min; (w, r)."""
    M = np.stack([_on_axis(genes, *reference_ctrl[ln]) for ln in lines])
    ok = np.isfinite(M).all(0)
    C = np.log1p(1e6 * M[:, ok] / np.nansum(M[:, ok], 1, keepdims=True))
    tt = np.log1p(np.asarray(target_cpm, np.float64)[ok])
    top = np.argsort(-C.var(0))[:ntop]
    r = np.array([np.corrcoef(C[i, top], tt[top])[0, 1] for i in range(len(lines))])
    zz = (r - r.mean()) / float(guards.floor(r.std(), sd_min, "order_sd_min"))
    w = np.exp((zz - zz.max()) / tau)
    return w / w.sum(), r


def cap_weights(w, cap):
    """Weights capped at `cap`, the excess spread over the uncapped lines in proportion."""
    w = np.asarray(w, np.float64) / np.sum(w)
    capped = False
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        ex = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        if capped := not free.any():  # fewer than 1/cap lines: all sit at the cap
            break
        w[free] += ex * w[free] / w[free].sum()
    guards.fired("cap_weights_all_capped", capped, 1)
    return w / w.sum() if capped else w


def plan_z(mix_z, mix_lines, reference_ctrl, genes, mean_cpm, p, g) -> np.ndarray:
    """|sum_t W_t z_t| / sqrt(sum_t W_t^2) over every line, W = u/T + (1 - u) capped similarity weight."""
    w = similarity_weights(genes, mean_cpm, reference_ctrl, mix_lines, p.sim_tau, p.sim_ntop, g.order_sd_min)[0]
    w = cap_weights(w, p.sim_cap)
    W = np.asarray(p.plan_z_uniform / len(w) + (1 - p.plan_z_uniform) * w, np.float64)
    guards.share(mix_lines, W, "plan", g)
    W = np.broadcast_to(W[:, None], mix_z.shape[:2])
    num = np.einsum("tg,tgp->gp", W, mix_z.astype(np.float64), optimize=True)
    return np.abs(num / np.sqrt((W**2).sum(0))[:, None])


def similarity_z(sim_z, sim_meas, sim_lines, reference_ctrl, genes, mean_cpm, p, g) -> np.ndarray:
    """sum_l w_l z_l / sqrt(sum_l w_l^2) over the lines measuring (g, p), capped similarity weights."""
    w = similarity_weights(genes, mean_cpm, reference_ctrl, sim_lines, p.sim_tau, p.sim_ntop, g.order_sd_min)[0]
    w = cap_weights(w, p.sim_cap)
    guards.share(sim_lines, w, "sim", g)
    S = np.zeros(sim_z.shape[1:])
    Q = np.zeros(sim_z.shape[1:])
    for t, wt in enumerate(w):
        wt = float(wt)
        if wt > 0:
            S += wt * sim_z[t]
            Q += (wt * wt) * sim_meas[t]
    return np.where(Q > 0, S / np.sqrt(np.maximum(Q, 1e-300)), 0.0)


def x_sim(rel, lines, reference_ctrl, genes, mean_cpm, p, g) -> np.ndarray:
    """(P,) training-data breadth averaged over the training-cell-line lines with capped softmax(xsim_tau) similarity weights."""
    keep = [i for i, ln in enumerate(lines) if ln in reference_ctrl]
    w, _ = similarity_weights(
        genes,
        np.asarray(mean_cpm, np.float64),
        reference_ctrl,
        [lines[i] for i in keep],
        p.xsim_tau,
        p.sim_ntop,
        g.order_sd_min,
    )
    w = cap_weights(w, p.sim_cap)
    guards.share([lines[i] for i in keep], w, "xsim", g)
    R = rel[keep]
    fin = np.isfinite(R)
    W = np.where(fin, w[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xs = np.where(W.sum(0) > 0, (np.where(fin, R, 0.0) * W).sum(0) / np.maximum(W.sum(0), 1e-300), np.nan)
        x = np.nanmean(rel, axis=0)
    x = np.where(np.isfinite(x), x, np.nanmedian(x))
    return np.where(np.isfinite(xs), xs, x)


# ------------------------------------------------------------------ sizes
def curve_sizes(breadth, eligible_count, m, median_degs) -> np.ndarray:
    """exp(slope (x - median x)) scaled to median median_degs, rounded, in [n_min, min(n_max, eligible genes)]."""
    x = np.asarray(breadth, np.float64)
    x = np.where(np.isfinite(x), x, np.nanmedian(x))
    n = np.exp(m.plan.curve_slope * (x - np.median(x)))
    if np.median(n) > 0:
        n = n * (median_degs / np.median(n))
    return np.clip(np.rint(n), m.calls.n_min, np.minimum(m.calls.n_max, eligible_count)).astype(np.int64)


def _rankz(x) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0 or np.all(x == x[0]):
        return np.zeros(x.shape)
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    return (r - r.mean()) / r.std()


def power_score(l2, z, zmix, cpm, own, m) -> np.ndarray:
    """rz(log1p expected detectable genes) - t_pen rz(log1p #genes with |plan z| > z_n2)."""
    p = m.plan
    eligible = (cpm > m.calls.eligible_cpm)[:, None] & own
    mu = cpm[:, None] * p.power_depth / 1e6
    pw = ndtr(p.power_k * np.abs(l2) / np.sqrt(1 / np.maximum(mu, 1e-3) + p.power_phi) - p.power_c)
    soft = np.clip(1 - p.power_soft_z**2 / np.maximum(np.abs(z), 1e-3) ** 2, 0, 1)
    result = _rankz(np.log1p(np.where(eligible, pw * soft, 0.0).sum(0)))
    n2 = np.log1p((eligible & (np.abs(zmix) > p.power_z_n2)).sum(0))
    return result - p.power_t_pen * _rankz(n2)


def s1_sizes(K0, s, m) -> np.ndarray:
    """PS arm: the multiset reassigned by score, the bottom shrink_quantile below the median to shrink_factor of
    their size, the freed calls spread over the above-median sizes in proportion (<= n_max)."""
    n_max = m.calls.n_max
    K0 = np.asarray(K0, np.int64)
    if K0.size <= 1:
        return K0.copy()
    o = np.lexsort((np.arange(len(s)), s))
    K = np.empty_like(K0)
    K[o] = np.sort(K0)
    K = K.astype(float).copy()
    med = np.median(K)
    r = np.argsort(np.argsort(s, kind="stable"), kind="stable") / (len(K) - 1)
    lo = (r < m.plan.shrink_quantile) & (K < med)
    new = np.maximum(1, K[lo] * m.plan.shrink_factor)
    freed = (K[lo] - new).sum()
    K[lo] = new
    hi = np.zeros(len(K), bool)
    hi[np.argsort(K, kind="stable")[len(K) // 2 + 1 :]] = True
    hi &= K > med
    for _ in range(20):
        room = hi & (K < n_max)
        if freed <= 1e-9 or not room.any():
            break
        add = K[room] * freed / K[room].sum()
        new = np.minimum(K[room] + add, n_max)
        freed -= (new - K[room]).sum()
        K[room] = new
    return np.maximum(1, np.round(K)).astype(np.int64)


def _curve_on(x, level, slope, n_max):
    n = np.exp(slope * (x - np.median(x)))
    return np.clip(np.round(level * n / np.median(n)), 1, n_max).astype(np.int64)


def xsim_sizes(curve, xs, score, m) -> np.ndarray:
    """'addc': the S1 sizes, raised for the perturbations in the upper half of x_sim toward the curve on x_sim
    (slope solved so its total equals the flat curve's), each add <= the S1 size, all adds <= 20% of the S1 total."""
    curve = np.asarray(curve, np.int64)
    x = np.asarray(xs, float)
    level = float(np.median(curve))
    total = int(curve.sum())
    p, n_max = m.plan, m.calls.n_max
    lo_b, hi_b = 0.0, 2 * p.curve_slope
    for _ in range(60):
        b = (lo_b + hi_b) / 2
        if _curve_on(x, level, b, n_max).sum() > total:
            hi_b = b
        else:
            lo_b = b
    xsb = _curve_on(x, level, (lo_b + hi_b) / 2, n_max)
    s1 = s1_sizes(curve, np.asarray(score, float), m)
    q = np.argsort(np.argsort(x, kind="stable"), kind="stable") / (x.size - 1)
    tgt = np.minimum(np.maximum(s1, xsb), np.minimum(p.xsim_cap_factor * s1, n_max))
    add = np.where(q >= p.xsim_strength, np.maximum(tgt - s1, 0), 0).astype(float)
    room = p.xsim_max_total * s1.sum() - s1.sum()
    if add.sum() > room:
        add *= room / add.sum()
    return (s1 + np.floor(add)).astype(np.int64)


def plan_sizes(breadth, xs, l2, z, zplan, mean_cpm, tidx, m, median_degs) -> np.ndarray:
    """Per-perturbation plan sizes: the curve at median_degs, power-score reassigned and x_sim raised, at most the
    eligible genes other than the own target."""
    own = own_mask(np.shape(l2), tidx)
    elig = np.asarray(mean_cpm, np.float64) > m.calls.eligible_cpm
    capacity = (elig[:, None] & own).sum(axis=0)
    sizes = curve_sizes(breadth, capacity, m, median_degs)
    active = capacity > 0
    allocated = np.zeros(len(tidx), dtype=np.int64)
    if active.any():
        score = power_score(
            finite(l2)[:, active],
            finite(z)[:, active],
            finite(zplan)[:, active],
            np.asarray(mean_cpm, float),
            own[:, active],
            m,
        )
        allocated[active] = xsim_sizes(sizes[active], np.asarray(xs, np.float64)[active], score, m)
    return np.minimum(allocated, capacity)


# ------------------------------------------------------------------ membership
def select_calls(l2, z, cpm, genes, perts, depth, m, median_calls=None, sizes=None) -> dict:
    """{pert: idx (priority order), sign, priority, nudge_sign (G,), nudge_strength (G,), n_calls}; the call counts
    are `sizes`, else median_calls x exp(gamma x strength)."""
    c = m.calls
    l2 = np.asarray(l2, float)
    z = np.asarray(z, float).copy()
    z[~np.isfinite(z)] = 0.0
    G, P = l2.shape
    cpm = np.asarray(cpm, float)
    elig = cpm > c.eligible_cpm
    gidx = {g: i for i, g in enumerate(genes)}
    tidx = np.array([gidx.get(p, -1) for p in perts])
    if sizes is None:
        s_p = _pert_strength(z, l2, elig, tidx, cpm, depth, c)
        n = median_calls * np.exp(c.gamma * s_p)
        K = np.round(np.clip(n, c.n_min, min(c.n_max, int(elig.sum()) - 1))).astype(int)
    else:
        K = np.asarray(sizes, int)
    guards.fired("no_eligible_genes", int(((K < 0) | (K > G)).sum()), K.size)
    K = np.clip(K, 0, G)  # no eligible genes gives n_max = -1
    mu = cpm[:, None] * depth / 1e6
    pw = np.abs(l2) / np.sqrt(1.0 / np.maximum(mu, 1e-3) + c.phi)
    gs = z / (z[z > 0].std() + 1e-9) + c.w_pow * pw / (pw[pw > 0].std() + 1e-9)
    sg_all = np.sign(l2)
    out = {}
    for k, p in enumerate(perts):
        s = np.where(elig, gs[:, k], -np.inf)
        if tidx[k] >= 0:
            s[tidx[k]] = -np.inf
        o = np.argsort(-s, kind="stable")[: int(np.isfinite(s).sum())]
        n = int(min(K[k], o.size))
        idx = o[:n]
        sign = sg_all[idx, k].copy()
        sign[sign == 0] = 1.0
        strength = np.zeros(G)
        rest = o[n:]
        if rest.size:
            strength[rest] = 1.0 - (np.arange(rest.size) + 0.5) / rest.size
        out[p] = dict(
            idx=idx,
            sign=sign,
            priority=1.0 - np.arange(n) / max(n, 1),
            nudge_sign=sg_all[:, k].astype(np.int8),
            nudge_strength=strength,
            n_calls=n,
        )
    return out


def _std(x):
    x = np.asarray(x, float)
    return (x - np.median(x)) / (x.std() + 1e-9)


def _pert_strength(z, l2, elig, tidx, cpm, depth, c):
    """Standardized log count of z > strength_z genes weighted by their detection probability."""
    ze = np.where(elig[:, None], z, 0.0)
    le = np.where(elig[:, None], l2, 0.0)
    for k, t in enumerate(tidx):
        if t >= 0:
            ze[t, k] = 0.0
            le[t, k] = 0.0
    mu = np.asarray(cpm, float)[:, None] * depth / 1e6
    pz = c.strength_k * np.abs(le) / np.sqrt(1.0 / np.maximum(mu, 1e-3) + c.strength_phi)
    f = np.log1p((ndtr(pz - c.strength_c) * (ze > c.strength_z)).sum(0))
    return _std(1.0 * _std(f))


def sized_plan(K, l2, zplan, mean_cpm, genes, perts, depth, common, m) -> dict:
    """select_calls at sizes K; every eligible gene gets a nudge direction (the training-data common sign where l2 = 0)."""
    calls = select_calls(
        np.nan_to_num(l2), np.nan_to_num(zplan, nan=0, posinf=0, neginf=0), mean_cpm, genes, perts, depth, m, sizes=K
    )
    el = mean_cpm > m.calls.eligible_cpm
    plans = {}
    for q in perts:
        c = calls[q]
        ns = c["nudge_sign"].astype(float)
        plans[q] = dict(
            idx=c["idx"],
            sign=c["sign"],
            priority=c["priority"],
            nudge_sign=np.where(el & (ns == 0), common, ns).astype(np.int8) * el,
            nudge_strength=c["nudge_strength"],
        )
    return plans
