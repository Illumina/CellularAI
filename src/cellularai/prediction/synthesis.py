"""n_cells integer cells per perturbation at fixed per-gene totals T.

Per eligible gene (control mean CPM > eligible_cpm) the cells take stratified control-CPM quantiles under a shift, bisected
against the scorer's exact Wilcoxon z, with the latent quantiles a Gaussian copula to the cell's target library rank
at correlation r; r is searched so the mean per-cell CPM approaches the placement target mt (mt <= 0: r = 0).
Other genes are scattered over the cells in bursts. The kernels deliberately preserve floating-point
operation order and random-number draws because both determine the final integer counts.
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
import tempfile

import numpy as np
from numba import njit
from scipy.stats import norm, rankdata

from ..evaluation.de import ControlOrderStats
from . import exp2


class Engine:
    """Control-side state of one context: the scorer's control order statistics (float32 sorted per-cell CPMs, tie
    sums, mean CPM), eligibility, pooled gene fractions, median depth."""

    def __init__(self, ctrl: np.ndarray, m):
        self.m = m
        X = np.asarray(ctrl)
        stats = ControlOrderStats.from_control_counts(X)
        self.sv, self.tie_sum, self.mean_cpm = stats.sorted_values, stats.tie_sum, stats.mean_cpm
        self.eligible = self.mean_cpm > self.m.calls.eligible_cpm
        self.elig_idx = np.flatnonzero(self.eligible)
        self.fc = np.asarray(X.sum(axis=0), dtype=np.float64)
        self.fc = self.fc / self.fc.sum()
        self.depth = int(round(np.median(np.asarray(X.sum(axis=1), dtype=np.float64))))
        self.G = X.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log2(self.mean_cpm / (1e6 * self.fc))
        self.log2_cg = np.where(np.isfinite(v), v, 0.0)

    def design_frac(self, l2):
        f = self.fc * exp2(np.asarray(l2, dtype=np.float64), self.m.guards.log2_max)
        return f / f.sum()

    def guard_l2(self, l2, plan_idx, plan_sign):
        """Planned genes' realized lfc (l2 - log2_cg) takes the plan sign with magnitude >= min_lfc."""
        need = self.m.calls.min_lfc
        l2 = np.array(l2, dtype=np.float64, copy=True)
        idx = np.asarray(plan_idx, dtype=np.int64)
        sg = np.sign(np.asarray(plan_sign, dtype=np.float64))
        sg[sg == 0] = 1.0
        off = self.log2_cg[idx]
        bad = sg * (l2[idx] - off) < need
        l2[idx[bad]] = off[bad] + sg[bad] * need
        return l2

    def z_targets(self, plan_idx, plan_sign, plan_priority, nudge_sign, nudge_strength):
        z_plan = self.m.synthesis.z_plan
        zt = (
            np.sign(np.asarray(nudge_sign, dtype=np.float64)) * np.clip(nudge_strength, 0, 1) * self.m.synthesis.z_nudge
        )
        if len(plan_idx):
            plan_idx = np.asarray(plan_idx, dtype=np.int64)
            sg = np.sign(np.asarray(plan_sign, dtype=np.float64))
            sg[sg == 0] = 1.0
            pr = np.asarray(plan_priority, dtype=np.float64)
            pr = (rankdata(pr) - 1) / max(pr.size - 1, 1) if pr.size > 1 else np.ones(1)
            zt[plan_idx] = sg * (z_plan[0] + (z_plan[1] - z_plan[0]) * pr)
        return zt


def apportion(frac, S: int) -> np.ndarray:
    """Largest-remainder integer totals summing to S, proportional to frac."""
    f = np.maximum(np.asarray(frac, dtype=np.float64), 0.0)
    f = f / f.sum()
    raw = f * S
    T = np.floor(raw).astype(np.int64)
    r = int(S - T.sum())
    if r > 0:
        T[np.argsort(-(raw - T), kind="stable")[:r]] += 1
    return T


@njit
def _z_exact(x, Lm, s, tie_ctrl, y, ys):
    n = x.size
    m = s.size
    for c in range(n):
        y[c] = np.float32((1e6 * x[c]) / Lm[c])
    ys[:] = np.sort(y)
    U = 0.0
    adj = 0.0
    i = 0
    while i < n:
        v = ys[i]
        j = i
        while j + 1 < n and ys[j + 1] == v:
            j += 1
        c = float(j - i + 1)
        lo = np.searchsorted(s, v, side="left")
        hi = np.searchsorted(s, v, side="right")
        U += c * (lo + 0.5 * (hi - lo))
        cc = float(hi - lo)
        mg = cc + c
        adj += (mg * mg * mg - mg) - (cc * cc * cc - cc)
        i = j + 1
    N = float(m + n)
    tc = 1.0 - (tie_ctrl + adj) / (N * N * N - N)
    var = tc * n * m * (N + 1.0) / 12.0
    if var <= 1e-24:
        return 0.0
    return (U - n * m / 2.0) / math.sqrt(var)


@njit
def _build(q, ordq, s, Lm, T, a, ntop, x):
    n = q.size
    m = s.size
    mass = 0
    for c in range(n):
        qq = a + (q[c] - 0.5)
        if qq <= 0.0:
            v = 0.0
        else:
            idx = int(qq * m)
            if idx >= m:
                idx = m - 1
            v = s[idx]
        xc = int(math.floor(v * Lm[c] * 1e-6 + 0.5))
        x[c] = xc
        mass += xc
    if mass < T:
        D = T - mass
        nd = ntop if D > ntop else D
        if nd > n:
            nd = n
        if nd < 1:
            nd = 1
        add = D // nd
        r = D - add * nd
        for j in range(nd):
            x[ordq[j]] += add + (1 if j < r else 0)
    elif mass > T:
        E = mass - T
        order = np.argsort(-x)
        cum = 0
        for i in range(n):
            cum += x[order[i]]
            nxt = x[order[i + 1]] if i + 1 < n else 0
            if cum - (i + 1) * nxt >= E:
                keep = cum - E
                h = keep // (i + 1)
                r = keep - h * (i + 1)
                for j in range(i + 1):
                    x[order[j]] = h + (1 if j < r else 0)
                break


@njit
def _build_b(ordq, T, w, x):
    n = x.size
    nz = n - int(w * n + 0.5)
    if nz < 1:
        nz = 1
    h = T // nz
    r = T - h * nz
    x[:] = 0
    for j in range(nz):
        x[ordq[j]] = h + (1 if j < r else 0)


@njit
def _solve_one(s, tie, q, ordq, Lm, T, target, kp, x, y, ys, best, xo):
    ntop, tol, max_it, lo, hi, wstep = kp
    _build(q, ordq, s, Lm, T, lo, ntop, x)
    zlo = _z_exact(x, Lm, s, tie, y, ys)
    if target <= zlo:
        xo[:] = x
        return zlo
    _build(q, ordq, s, Lm, T, hi, ntop, x)
    zhi = _z_exact(x, Lm, s, tie, y, ys)
    if target >= zhi:
        xo[:] = x
        zres = zhi
        bz = zhi
        wprev = 0.0
        for ib in range(1, 39):
            w = wstep * ib
            _build_b(ordq, T, w, x)
            z = _z_exact(x, Lm, s, tie, y, ys)
            if z >= target:
                wlo, whi = wprev, w
                best[:] = x
                bz2 = z
                for it in range(12):
                    wm = 0.5 * (wlo + whi)
                    _build_b(ordq, T, wm, x)
                    zm = _z_exact(x, Lm, s, tie, y, ys)
                    if zm >= target:
                        whi = wm
                        best[:] = x
                        bz2 = zm
                    else:
                        wlo = wm
                    if abs(zm - target) < tol:
                        break
                xo[:] = best
                zres = bz2
                break
            if z > bz:
                bz = z
                xo[:] = x
                zres = z
            wprev = w
        return zres
    berr = 1e300
    bz = 0.0
    for it in range(max_it):
        mid = 0.5 * (lo + hi)
        _build(q, ordq, s, Lm, T, mid, ntop, x)
        z = _z_exact(x, Lm, s, tie, y, ys)
        e = abs(z - target)
        if e < berr:
            berr = e
            bz = z
            best[:] = x
        if e < tol:
            break
        if z < target:
            lo = mid
        else:
            hi = mid
    xo[:] = best
    return bz


@njit
def _q_from(e0, zl, r, q, ordq, e):
    n = e0.size
    c = math.sqrt(max(1.0 - r * r, 0.0))
    for i in range(n):
        e[i] = r * zl[i] + c * e0[i]
    o = np.argsort(e)
    for i in range(n):
        q[o[i]] = (i + 0.5) / n
        ordq[i] = o[n - 1 - i]


@njit
def _mcpm(x, Lm):
    n = x.size
    acc = 0.0
    for c in range(n):
        acc += 1e6 * x[c] / Lm[c]
    return acc / n


@njit
def _solve_block_place(
    sv, tie_sum, gidx, T, zt, E0, zl, Lm, mt, rprev, rmax, n_bis, kp, zp, ztolv, ltol, cpm0, out, rout, zlimv
):
    """Per gene: r = 0 solve, z repair for overshooting unplanned genes, else an r search toward mt with z held to
    the r = 0 accuracy + ztol. An unplanned gene whose realized mean lands on the other side of the control mean
    from its nudge z is demoted to rank-neutral (zt = 0, zlimv = -1)."""
    J = gidx.size
    n = E0.shape[1]
    tol = kp[1]
    planned_z, overshoot_z, ra, rb = zp
    x = np.empty(n, dtype=np.int64)
    y = np.empty(n, dtype=np.float32)
    ys = np.empty(n, dtype=np.float32)
    best = np.empty(n, dtype=np.int64)
    xc = np.empty(n, dtype=np.int64)
    xb = np.empty(n, dtype=np.int64)
    q = np.empty(n)
    ordq = np.empty(n, dtype=np.int64)
    e = np.empty(n)
    for j in range(J):
        g = gidx[j]
        s = sv[g]
        tie = tie_sum[g]
        e0 = E0[j]
        target = zt[j]
        _q_from(e0, zl, 0.0, q, ordq, e)
        z0 = _solve_one(s, tie, q, ordq, Lm, T[j], target, kp, x, y, ys, best, xb)
        m0 = _mcpm(xb, Lm)
        bm = m0
        br = 0.0
        zlim = max(abs(z0 - target), tol) + ztolv[j]
        zfail = abs(target) < planned_z and (abs(z0) - abs(target)) > overshoot_z
        if zfail:
            bez = abs(z0 - target)
            bgap = 1e300
            inband = False
            for ig in range(7):
                rg = (ra, rb, rmax, -ra, -rb, -rmax, 0.0)[ig]
                _q_from(e0, zl, rg, q, ordq, e)
                zc = _solve_one(s, tie, q, ordq, Lm, T[j], target, kp, x, y, ys, best, xc)
                mc = _mcpm(xc, Lm)
                ez = abs(zc - target)
                gap = abs(math.log2(mc / mt[j])) if (mt[j] > 0.0 and mc > 0.0) else 0.0
                if abs(zc - target) <= tol + ztolv[j]:
                    if (not inband) or gap < bgap:
                        inband = True
                        bgap = gap
                        bez = ez
                        bm = mc
                        br = rg
                        xb[:] = xc
                elif (not inband) and ez < bez:
                    bez = ez
                    bgap = gap
                    bm = mc
                    br = rg
                    xb[:] = xc
            zlim = max(bez, tol) + ztolv[j]
        elif mt[j] > 0.0 and m0 > 0.0:
            d0 = math.log2(m0 / mt[j])
            bd = abs(d0)
            if bd >= ltol:
                sgn = 1.0 if d0 > 0.0 else -1.0
                cand0 = rprev[j] if rprev[j] * sgn > 0.0 else 0.0
                rlo = 0.0
                rhi = sgn * rmax
                done = False
                if cand0 != 0.0:
                    _q_from(e0, zl, cand0, q, ordq, e)
                    zc = _solve_one(s, tie, q, ordq, Lm, T[j], target, kp, x, y, ys, best, xc)
                    mc = _mcpm(xc, Lm)
                    if mc > 0.0 and abs(zc - target) <= zlim:
                        dc = math.log2(mc / mt[j])
                        if abs(dc) < bd:
                            bd = abs(dc)
                            bm = mc
                            br = cand0
                            xb[:] = xc
                        if abs(dc) < ltol:
                            done = True
                        elif dc * sgn > 0.0:
                            rlo = cand0
                        else:
                            rhi = cand0
                    else:
                        rhi = cand0
                if not done:
                    if rhi == sgn * rmax:
                        _q_from(e0, zl, rhi, q, ordq, e)
                        zc = _solve_one(s, tie, q, ordq, Lm, T[j], target, kp, x, y, ys, best, xc)
                        mc = _mcpm(xc, Lm)
                        okz = mc > 0.0 and abs(zc - target) <= zlim
                        dc = math.log2(mc / mt[j]) if mc > 0.0 else 0.0
                        if okz and abs(dc) < bd:
                            bd = abs(dc)
                            bm = mc
                            br = rhi
                            xb[:] = xc
                        if okz and dc * sgn > 0.0:
                            done = True
                    for it in range(n_bis):
                        if done or bd < ltol:
                            break
                        rm = 0.5 * (rlo + rhi)
                        _q_from(e0, zl, rm, q, ordq, e)
                        zc = _solve_one(s, tie, q, ordq, Lm, T[j], target, kp, x, y, ys, best, xc)
                        mc = _mcpm(xc, Lm)
                        okz = mc > 0.0 and abs(zc - target) <= zlim
                        if not okz:
                            rhi = rm
                            continue
                        dc = math.log2(mc / mt[j])
                        if abs(dc) < bd:
                            bd = abs(dc)
                            bm = mc
                            br = rm
                            xb[:] = xc
                        if dc * sgn > 0.0:
                            rlo = rm
                        else:
                            rhi = rm
        out[j, :] = xb
        rout[j] = br
        zlimv[j] = zlim
        if target != 0.0 and abs(target) < planned_z and mt[j] > 0.0 and bm > 0.0 and (bm - cpm0[j]) * target < 0.0:
            zt[j] = 0.0
            zlimv[j] = -1.0


def synthesize(eng: Engine, job: dict) -> np.ndarray:
    """(n_cells, G) float32 integer counts of one perturbation; n_cells is the length of its target libraries."""
    rng, sy = job["rng"], eng.m.synthesis
    n = len(job["Lt"])
    T = np.asarray(job["T"], dtype=np.int64)
    zt_all = eng.z_targets(
        job["plan_idx"], job["plan_sign"], job["plan_priority"], job["nudge_sign"], job["nudge_strength"]
    )
    # The library permutation is drawn from rng's current state without advancing it.
    st = rng.bit_generator.state["state"]["state"]
    Lt = np.random.default_rng(int(st) % (2**63)).permutation(np.asarray(job["Lt"], dtype=np.float64))
    if int(round(Lt.sum())) != int(T.sum()):
        raise ValueError(f"target libraries sum {Lt.sum():.6g} != the gene totals' {int(T.sum())}")
    X = np.zeros((n, eng.G), dtype=np.float32)
    ne = np.flatnonzero(~eng.eligible & (T > 0))
    p = Lt / Lt.sum()
    if ne.size:
        rng.choice(n, size=int(T[ne].sum()), p=p)  # Required draw in the established random sequence.
        st2 = rng.bit_generator.state["state"]["state"]
        r2 = np.random.default_rng((int(st2) + 1) % (2**63))
        nb = sy.ne_burst
        Tn = T[ne].astype(np.int64)
        nbur = -(-Tn // nb)
        gid = np.repeat(np.arange(ne.size), nbur)
        j = np.arange(gid.size) - np.repeat(np.cumsum(nbur) - nbur, nbur)
        base = Tn[gid] // nbur[gid]
        size = base + (j < Tn[gid] - base * nbur[gid])
        c2 = r2.choice(n, size=gid.size, p=p)
        cnt = np.bincount(c2 * ne.size + gid, weights=size, minlength=n * ne.size)
        X[:, ne] = cnt.reshape(n, ne.size)
    gi = eng.elig_idx
    J = gi.size
    eps = rng.standard_normal((J, n))
    Tg = T[gi].astype(np.int64)
    zt = zt_all[gi].astype(np.float64)
    out = np.zeros((J, n), dtype=np.int64)
    mt = np.asarray(job["mt"], dtype=np.float64)[gi]
    rout = np.zeros(J)
    zlimv = np.zeros(J)
    ranks = np.argsort(np.argsort(Lt, kind="stable"), kind="stable")
    zl = norm.ppf((ranks + 0.5) / n)
    cpm0 = np.asarray(eng.mean_cpm, dtype=np.float64)[gi].copy()
    zt0 = zt.copy()
    ztolv = np.full(J, sy.ztol)
    ztolv[np.abs(zt0) >= sy.planned_z] = sy.ztol_plan
    kp = (
        int(sy.solve_ntop),
        float(sy.solve_tol),
        int(sy.solve_iters),
        *map(float, sy.shift_band),
        float(sy.sparsify_step),
    )
    args = (sy.rmax, sy.n_bis, kp, (float(sy.planned_z), float(sy.overshoot_z), *map(float, sy.repair_r)))
    _solve_block_place(
        eng.sv,
        eng.tie_sum,
        gi,
        Tg,
        zt,
        eps,
        zl,
        np.asarray(Lt, np.float64).copy(),
        mt,
        np.zeros(J),
        *args,
        ztolv,
        sy.ltol,
        cpm0,
        out,
        rout,
        zlimv,
    )
    X[:, gi] = out.T
    Lm = np.maximum(X.sum(axis=1, dtype=np.float64), 1.0)
    # Genes demoted at their solve are re-solved once rank-neutral at the realized libraries.
    dem = np.flatnonzero((zlimv == -1.0) & (zt == 0.0) & (zt0 != 0.0))
    if dem.size:
        o2 = out[dem].copy()
        _solve_block_place(
            eng.sv,
            eng.tie_sum,
            gi[dem],
            Tg[dem],
            zt[dem].copy(),
            eps[dem],
            zl,
            Lm,
            mt[dem],
            rout[dem].copy(),
            *args,
            ztolv[dem],
            sy.ltol,
            cpm0[dem],
            o2,
            rout[dem].copy(),
            zlimv[dem].copy(),
        )
        out[dem] = o2
        X[:, gi] = out.T
    return X


_ENG: list = []


def _init(eng: Engine, sv_path: str) -> None:
    eng.sv = np.asarray(np.load(sv_path, mmap_mode="r"))
    _ENG[:] = [eng]


def _worker(job: dict) -> np.ndarray:
    return synthesize(_ENG[0], job)


def synthesize_all(eng: Engine, jobs: list[dict], n_jobs: int) -> list[np.ndarray]:
    """One block per job, in job order. Workers are forkserver processes given the engine once, its control order
    statistics (> 1 GB at 18k cells) as one read-only memory map all share, and each job by value: nothing is
    inherited (fork after the pyarrow/numba threads exist may deadlock)."""
    if n_jobs <= 1:
        return [synthesize(eng, j) for j in jobs]
    temporary = os.environ.get("TMPDIR") or ("/dev/shm" if os.path.isdir("/dev/shm") else None)
    with tempfile.TemporaryDirectory(dir=temporary) as d:
        np.save(f"{d}/sv.npy", eng.sv)
        light = copy.copy(eng)
        light.sv = None
        with mp.get_context("forkserver").Pool(
            min(n_jobs, len(jobs)), initializer=_init, initargs=(light, f"{d}/sv.npy")
        ) as pool:
            return pool.map(_worker, jobs, chunksize=1)
