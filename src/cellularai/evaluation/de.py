"""Wilcoxon DE tables against one control group's precomputed order statistics.

Conventions: per-cell CPM (totals floored at 1),
genes eligible when the control mean CPM > 5, normal approximation without continuity correction, pooled
tie correction, all-tied gene -> p = 1, BH within the perturbation over eligible genes, alpha 0.05,
lfc = log2((pert mean CPM + 1e-9) / (control mean CPM + 1e-9)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import false_discovery_control, norm

MIN_CTRL_CPM = 5.0
ALPHA = 0.05
LFC_EPS = 1e-9


@dataclass
class ControlOrderStats:
    sorted_values: np.ndarray  # (G, m) float32 per-cell CPM, each row ascending
    tie_sum: np.ndarray  # (G,) sum of t^3 - t over the control's value multiplicities
    mean_cpm: np.ndarray  # (G,)

    @classmethod
    def from_control_counts(cls, control_counts: np.ndarray) -> "ControlOrderStats":
        counts = np.asarray(control_counts)
        m, G = counts.shape
        cpm_sum = np.zeros(G, dtype=np.float64)
        sorted_values = np.empty((G, m), dtype=np.float32)
        chunk = 4096
        for start in range(0, m, chunk):
            dense = np.asarray(counts[start : start + chunk]).astype(np.float64, copy=False)
            tot = np.maximum(dense.sum(axis=1), 1.0)
            cpm = 1e6 * dense / tot[:, None]
            cpm_sum += cpm.sum(axis=0)
            sorted_values[:, start : start + dense.shape[0]] = cpm.T.astype(np.float32)
        sorted_values.sort(axis=1)
        tie_sum = np.zeros(G, dtype=np.float64)
        for g in range(G):
            edges = np.concatenate(([0], np.flatnonzero(np.diff(sorted_values[g])) + 1, [m]))
            cnts = np.diff(edges).astype(np.float64)
            tie_sum[g] = float(np.sum(cnts**3 - cnts))
        return cls(sorted_values, tie_sum, cpm_sum / m)


def de_table(pert_counts: np.ndarray, stats: ControlOrderStats) -> dict:
    """pval / padj / lfc on the full gene axis (NaN where ineligible) plus significant / eligible masks."""
    pert_counts = np.asarray(pert_counts, dtype=np.float64)
    n, G = pert_counts.shape
    totals = np.maximum(pert_counts.sum(axis=1), 1.0)
    pert_cpm = 1e6 * pert_counts / totals[:, None]

    eligible = stats.mean_cpm > MIN_CTRL_CPM
    elig_idx = np.flatnonzero(eligible)
    pvals = np.full(G, np.nan)
    padj = np.full(G, np.nan)
    lfc = np.full(G, np.nan)
    significant = np.zeros(G, dtype=bool)
    if elig_idx.size == 0:
        return {"pval": pvals, "padj": padj, "lfc": lfc, "significant": significant, "eligible": eligible}

    # Rank on the control side's float32 grid so ties with control values are seen as ties.
    vals32 = pert_cpm[:, elig_idx].T.astype(np.float32)
    U = np.empty(elig_idx.size, dtype=np.float64)
    pooled_tie = np.empty(elig_idx.size, dtype=np.float64)
    for i, g in enumerate(elig_idx):
        row = stats.sorted_values[g]
        v = vals32[i]
        lo = np.searchsorted(row, v, side="left")
        hi = np.searchsorted(row, v, side="right")
        U[i] = lo.sum() + 0.5 * (hi - lo).sum()
        d, c = np.unique(v, return_counts=True)
        c_ctrl = (np.searchsorted(row, d, side="right") - np.searchsorted(row, d, side="left")).astype(np.float64)
        merged = c_ctrl + c.astype(np.float64)
        pooled_tie[i] = stats.tie_sum[g] - np.sum(c_ctrl**3 - c_ctrl) + np.sum(merged**3 - merged)

    m = float(stats.sorted_values.shape[1])
    nn = float(n)
    total = m + nn
    tc = 1.0 - pooled_tie / (total**3 - total)
    sd = np.sqrt(np.maximum(tc * nn * m * (total + 1.0) / 12.0, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (U - nn * m / 2.0) / sd
    tie_floor = sd < 1e-12
    p = np.where(tie_floor, 1.0, 2.0 * norm.sf(np.abs(np.where(tie_floor, 0.0, z))))

    pvals[elig_idx] = p
    padj[elig_idx] = false_discovery_control(p, method="bh")
    pert_mean = pert_cpm[:, elig_idx].mean(axis=0)
    lfc[elig_idx] = np.log2((pert_mean + LFC_EPS) / (stats.mean_cpm[elig_idx] + LFC_EPS))
    significant[elig_idx] = padj[elig_idx] < ALPHA
    return {"pval": pvals, "padj": padj, "lfc": lfc, "significant": significant, "eligible": eligible}
