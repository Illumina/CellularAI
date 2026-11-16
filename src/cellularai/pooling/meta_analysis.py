"""One weighted pass over every corrected summary row for both streams.

For stream s, row t, gene g, perturbation p the weight is ``row_s[t, g] * col_s[t, p] * mat_s[t]``,
each op a ``*``-product of named factors. log2FC pools as a weighted mean (an unmeasured entry
counts as 0 at its row's weight); z pools as Liptak's ``|sum w z| / sqrt(sum w^2)`` over the
measured entries. The configured operators and their multiplication order are part of the model definition.
"""

from __future__ import annotations

import warnings

import numpy as np

from .. import guards
from .catalog import VARIANCE_FLOOR, OffsetTable, apply_offsets, signed_z
from .weights import CellLineFactors

AXES = ("row", "col", "mat")
CELL_LINE = frozenset(n for names in CellLineFactors.ORDER.values() for n in names)
# Factors computed here rather than on the factors directory.
LOCAL = {"row": ("inv:mad:dl",), "col": (), "mat": ("auxiliary",)}
# The pooled-effect boundary takes log2(max(2**u, 1e-12)): identity above this bound, 0 for non-finite u.
_L2_FLOOR = float(np.log2(1e-12))


def to_l2(u: np.ndarray) -> np.ndarray:
    out = np.where(np.isfinite(u), u, 0.0)
    return np.maximum(out, _L2_FLOOR).astype(np.float64)


def parse_op(expr: str, axis: str) -> list[str]:
    """``'sef*kd'`` -> ``['sef', 'kd']``; ``'1'`` -> ``[]``. A factor outside the supported set raises."""
    known = set(LOCAL[axis]) | set(CellLineFactors.ORDER[axis])
    names = [n.strip() for n in str(expr).split("*") if n.strip() not in ("", "1")]
    if bad := [n for n in names if n not in known]:
        raise ValueError(f"{axis} op {expr!r}: {bad} not supported; known {sorted(known)}")
    if len(set(names)) != len(names):
        raise ValueError(f"{axis} op {expr!r} names a factor twice")
    return names


def table_field(tables: list[OffsetTable], genes, name: str) -> np.ndarray:
    """(T, G) one per-gene field of each row's table on `genes`, NaN where absent."""
    G = len(genes)
    out = np.full((len(tables), G), np.nan)
    for t, tab in enumerate(tables):
        pos = {g: i for i, g in enumerate(tab.genes)}
        idx = np.fromiter((pos.get(str(g), -1) for g in genes), np.int64, G)
        ok = idx >= 0
        out[t, ok] = np.asarray(getattr(tab, name), dtype=np.float64)[idx[ok]]
    return out


def with_fallback(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(V with every non-estimate filled, m): a (row, gene) without a finite positive estimate
    borrows the gene's median over the rows that have one; a gene no row estimates takes the
    median of those medians. ``m`` is that gene median."""
    ok = np.isfinite(V) & (V > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmedian(np.where(ok, V, np.nan), axis=0)
    fin = m[np.isfinite(m)]
    g = np.median(fin) if fin.size else 1.0
    m = np.where(np.isfinite(m), m, g)
    return np.where(ok, V, m[None, :]), m


def dl_tau2(w: np.ndarray, centred, T: int, G: int, P: int) -> np.ndarray:
    """(G,) DerSimonian-Laird between-row variance per gene on the centred log2FC.

    Over the rows measuring (g, p): Q = sum w (L - Lbar_w)^2, df = k - 1, C = sum w - sum w^2 / sum w,
    each summed over p; tau2 = max(0, (Q - df) / C).
    """
    S0, S1, S2, SQ, K = (np.zeros((G, P)) for _ in range(5))
    for t in range(T):
        L_raw, L = centred(t)
        m = np.isfinite(L_raw) & np.isfinite(L)
        wt = np.where(m, w[t][:, None], 0.0)
        Lm = np.where(m, L, 0.0)
        S0 += wt
        S1 += wt * Lm
        S2 += wt * Lm * Lm
        SQ += wt * wt
        K += m
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(S0 > 0, S2 - S1 * S1 / S0, 0.0)
        cc = np.where(S0 > 0, S0 - SQ / S0, 0.0)
    Q, C = q.sum(axis=1), cc.sum(axis=1)
    df = np.maximum(K - 1.0, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tau2 = np.where(C > 0, (Q - df) / np.maximum(C, 1e-300), 0.0)
    return np.maximum(tau2, 0.0)


def inv_mad_dl(tables, genes, centred, T: int, P: int, v_floor_rel: float) -> np.ndarray:
    """(T, G) ``m_g / (sigma2_tg + tau2_g)``: sigma2 the row's full-axis null variance, tau2 by DL; counts the
    variances below v_floor_rel m_g (a row weighing over 1 / v_floor_rel times the median row) and floors them there."""
    V, m = with_fallback(table_field(tables, genes, "response_variance"))
    m = m[None, :]
    V = V + dl_tau2(1.0 / V, centred, T, len(genes), P)[None, :]
    guards.under(V / m, v_floor_rel, "v_floor_rel")
    return m / np.maximum(V, v_floor_rel * m)


def assert_floor_not_binding(W: np.ndarray) -> None:
    """A weight near 1 / VARIANCE_FLOOR is a collapsed variance that would own the pool."""
    top = float(np.nanmax(W)) if np.isfinite(W).any() else 0.0
    if top >= 1.0 / (100.0 * VARIANCE_FLOOR):
        raise ValueError(f"pooling weight {top:.3e} is within 100x of the VARIANCE_FLOOR ceiling")


def combine_layer(
    rows_l2, rows_p, tables, genes, ops: dict, multipliers, v_floor_rel: float, factors: CellLineFactors | None = None
):
    """Centre every row (log2FC and z on their own offsets, z signed by the raw log2FC), then pool.

    Row order is the caller's and fixed: the sums accumulate along it and float addition is not
    associative. Returns the layer's (l2, z).
    """
    G, P = rows_l2[0].shape
    T = len(rows_l2)
    M = np.asarray(multipliers, dtype=np.float64)
    parsed = {s: {ax: parse_op(ops[f"{s}_{ax}"], ax) for ax in AXES} for s in ("l2", "z")}
    dn = {s: {n for ax in AXES for n in parsed[s][ax] if n in CELL_LINE} for s in parsed}
    if (dn["l2"] or dn["z"]) and factors is None:
        raise ValueError(f"ops name factors factor(s) {sorted(dn['l2'] | dn['z'])}; pass `factors`")

    def corrected(t):
        L_raw = np.asarray(rows_l2[t], dtype=np.float64)
        p = np.asarray(rows_p[t], dtype=np.float64)
        L = apply_offsets(L_raw, genes, tables[t], "l2")
        return L_raw, L, apply_offsets(signed_z(L_raw, p), genes, tables[t], "z")

    def centred(t):
        L_raw = np.asarray(rows_l2[t], dtype=np.float64)
        return L_raw, apply_offsets(L_raw, genes, tables[t], "l2")

    memo: dict = {}

    def factor(n):
        if n not in memo:
            memo[n] = M if n == "auxiliary" else inv_mad_dl(tables, genes, centred, T, P, v_floor_rel)
        return memo[n]

    def pipeline(stream):
        o = parsed[stream]
        R = None
        for n in o["row"]:
            if n not in CELL_LINE:
                R = factor(n) if R is None else R * factor(n)
        R = np.ones((T, G)) if R is None else R
        Mt = None
        for n in o["mat"]:
            if n not in CELL_LINE:
                Mt = factor(n) if Mt is None else Mt * factor(n)
        return R if Mt is None else R * Mt[:, None]

    W0, W0_z = pipeline("l2"), pipeline("z")
    assert_floor_not_binding(W0)

    num_l, den_l, num_z, sqw_z = (np.zeros((G, P)) for _ in range(4))
    for t in range(T):
        L_raw, L, z = corrected(t)
        measured = np.isfinite(L_raw)
        w = np.broadcast_to(W0[t][:, None], (G, P)).copy()
        w_z = np.broadcast_to(W0_z[t][:, None], (G, P)).copy()
        e = factors.block(t, dn["l2"]) if dn["l2"] else None
        if e is not None:
            w = w * e
        if dn["z"]:
            e_z = e if dn["z"] == dn["l2"] else factors.block(t, dn["z"])
            if e_z is not None:
                w_z = w_z * e_z
        Lf = np.where(measured, L, 0.0)
        zf = np.where(np.isfinite(z), z, 0.0)
        wz = np.where(np.isfinite(z), w_z, 0.0)
        num_l += w * Lf
        den_l += w
        num_z += wz * zf
        sqw_z += wz * wz

    with np.errstate(invalid="ignore", divide="ignore"):
        l2 = np.where(den_l > 0, num_l / np.maximum(den_l, 1e-300), np.nan)
        z = np.where(sqw_z > 0, np.abs(num_z) / np.sqrt(np.maximum(sqw_z, 1e-300)), 0.0)
    return to_l2(l2), np.where(np.isfinite(z), z, 0.0)
