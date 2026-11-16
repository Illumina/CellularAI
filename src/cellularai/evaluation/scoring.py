"""The VCC metric suite: six raw metrics, anchor scaling and overall score.

The metrics compare a prediction side and a reference side, each a `SideStats` built from one
(cells, controls) pair. The reference side is cached with the anchors; only the prediction side is built.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

from .de import ControlOrderStats, de_table

TS = 5e4  # profile normalizer
PURITY_THRESHOLD = 0.9  # reach: minimum directional purity
NMAE_MIN_GATE = 10  # nmae: minimum reference-gate size
METRIC_NAMES = ("pds", "expr_mse", "fid", "reach", "jac", "nmae")


def group_profile(counts: np.ndarray) -> np.ndarray:
    """Group-summed, TS-normalized, log1p profile."""
    counts = np.asarray(counts, dtype=np.float64)
    return np.log1p(TS * counts.sum(axis=0) / max(counts.sum(), 1e-12))


def jackknife_moment(counts: np.ndarray) -> float:
    """The expr_mse delete-one jackknife C_p = ((n-1)/n) sum_g sum_i (v_ig - mean_g)^2,
    streamed as sum v^2 - (sum v)^2/n."""
    counts = np.asarray(counts)
    n = counts.shape[0]
    if n < 2:
        return 0.0
    P = np.asarray(counts.sum(axis=0)).ravel().astype(np.float64)
    S = float(P.sum())
    acc_sq = np.zeros(P.shape[0], dtype=np.float64)
    acc_sum = np.zeros(P.shape[0], dtype=np.float64)
    for start in range(0, n, 512):
        dense = np.asarray(counts[start : start + 512]).astype(np.float64, copy=False)
        remaining = np.maximum(S - dense.sum(axis=1, keepdims=True), 1e-12)
        v = np.log1p(TS * (P[None, :] - dense) / remaining)
        acc_sq += (v * v).sum(axis=0)
        acc_sum += v.sum(axis=0)
    return float((n - 1) / n * (acc_sq - acc_sum**2 / n).sum())


@dataclass
class SideStats:
    perts: list[str]
    genes: list[str]
    profiles: Dict[str, np.ndarray]
    control_profile: np.ndarray
    de: Dict[str, dict]
    jackknife: Dict[str, float] = field(default_factory=dict)
    control_jackknife: float = 0.0

    def target_index(self, pert: str) -> int:
        try:
            return self.genes.index(pert)
        except ValueError:
            return -1

    @classmethod
    def load(cls, path) -> "SideStats":
        """Load a cached reference-side summary archive."""
        z = np.load(path, allow_pickle=False)
        perts = [str(p) for p in z["perts"]]
        de = {}
        for i, p in enumerate(perts):
            if f"de{i}_pval" in z:
                de[p] = {k: z[f"de{i}_{k}"] for k in ("pval", "padj", "lfc", "significant", "eligible")}
        return cls(
            perts=perts,
            genes=[str(g) for g in z["genes"]],
            profiles={p: z[f"profile_{i}"] for i, p in enumerate(perts)},
            control_profile=z["control_profile"],
            de=de,
            jackknife={str(p): float(v) for p, v in zip(z["jackknife_perts"], z["jackknife_vals"])},
            control_jackknife=float(z["control_jackknife"]),
        )


# Fork workers inherit these by copy-on-write; the control order stats are too large to pickle per task.
_CTX: dict = {}


def _side_worker(pert: str):
    X = np.asarray(_CTX["cells"][pert])
    return group_profile(X), de_table(X, _CTX["order"]), jackknife_moment(X)


def build_prediction_side(
    cells: Dict[str, np.ndarray], controls: np.ndarray, genes: Sequence[str], perts: Sequence[str], n_jobs: int = 1
) -> SideStats:
    """Profiles, DE tables and jackknife moments of predicted cells against the control provision.

    The prediction side's control jackknife is not built: expr_mse reads only the reference's.
    """
    genes, perts = list(genes), list(perts)
    controls = np.asarray(controls)
    _CTX.update(cells=cells, order=ControlOrderStats.from_control_counts(controls))
    try:
        if n_jobs > 1:
            with mp.get_context("fork").Pool(processes=min(n_jobs, len(perts))) as pool:
                results = pool.map(_side_worker, perts, chunksize=1)
        else:
            results = [_side_worker(p) for p in perts]
    finally:
        _CTX.clear()
    return SideStats(
        perts=perts,
        genes=genes,
        profiles={p: r[0] for p, r in zip(perts, results)},
        control_profile=group_profile(controls),
        de={p: r[1] for p, r in zip(perts, results)},
        jackknife={p: r[2] for p, r in zip(perts, results)},
    )


def _shared(pred: SideStats, ref: SideStats) -> list[str]:
    return [p for p in pred.perts if p in ref.profiles]


def pds_cosine_panel(pred: SideStats, ref: SideStats, panel_targets) -> float:
    """pds: profile distances with the whole evaluation panel's target genes removed; zero-norm effects sit at
    distance 1."""
    shared = _shared(pred, ref)
    if len(shared) < 2:
        return float("nan")
    G = pred.control_profile.shape[0]
    keep = np.ones(G, dtype=bool)
    keep[np.asarray(sorted({t for t in panel_targets if 0 <= t < G}), dtype=np.int64)] = False
    names = sorted(shared)
    n = len(names)
    P = np.stack([(pred.profiles[p] - pred.control_profile)[keep] for p in names])
    R = np.stack([(ref.profiles[p] - ref.control_profile)[keep] for p in names])
    pn = np.linalg.norm(P, axis=1)
    rn = np.linalg.norm(R, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = 1.0 - (P @ R.T) / np.outer(pn, rn)
    D[(pn[:, None] == 0) | (rn[None, :] == 0)] = 1.0
    D = np.clip(D, 0.0, 2.0)
    scores = np.empty(n)
    for i in range(n):
        row, own = D[i], D[i, i]
        k = int(np.sum(row < own)) + 0.5 * max(int(np.sum(row == own)) - 1, 0)
        scores[i] = 1.0 - k / (n - 1)
    return float(scores.mean())


def expr_mse_parts(pred: SideStats, ref: SideStats) -> Dict[str, float]:
    """expr_mse (unbiased, capped, normalized squared error) and its pieces.

    N_p = (1/G_p)(|b^_p - b_p|^2 - rho min(C^_p, C_p) - C_p), D_p = (1/G_p)(|b_p - b_ctrl|^2 - C_p - C_ctrl),
    distances over every gene but p's own target; rho = min(1, spread_B / credit) with
    credit = sum_p (1/G_p) min(C^_p, C_p) and spread_B the 1/G_p-weighted across-perturbation spread.
    """
    shared = _shared(pred, ref)
    if not shared:
        return {}
    G = pred.control_profile.shape[0]
    C_hat = np.array([pred.jackknife.get(p, 0.0) for p in shared])
    C_ref = np.array([ref.jackknife.get(p, 0.0) for p in shared])
    G_vec = np.array([G - 1 if ref.target_index(p) >= 0 else G for p in shared], dtype=np.float64)
    credit = (1.0 / G_vec) * np.minimum(C_hat, C_ref)

    P = np.stack([pred.profiles[p] for p in shared])
    w = 1.0 / G_vec
    ok = np.ones(P.shape, dtype=bool)
    for i, p in enumerate(shared):
        t = pred.target_index(p)
        if 0 <= t < G:
            ok[i, t] = False
    wmat = np.where(ok, w[:, None], 0.0)
    wsum = wmat.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        wmean = np.where(wsum > 0, (wmat * P).sum(axis=0) / wsum, 0.0)
    dev = np.where(ok, P - wmean[None, :], 0.0)
    spread_B = float((w[:, None] * dev * dev).sum())
    cr = float(credit.sum())
    rho = 1.0 if cr <= 0 else float(min(1.0, spread_B / cr))

    N_total = D_total = dist = 0.0
    for i, p in enumerate(shared):
        t = ref.target_index(p)
        pred_m, ref_m, ctrl_m = pred.profiles[p], ref.profiles[p], ref.control_profile
        if 0 <= t < G:
            mask = np.ones(G, dtype=bool)
            mask[t] = False
            pred_m, ref_m, ctrl_m = pred_m[mask], ref_m[mask], ctrl_m[mask]
        d2 = np.sum((pred_m - ref_m) ** 2)
        dist += d2 / G_vec[i]
        N_total += (d2 - rho * min(C_hat[i], C_ref[i]) - C_ref[i]) / G_vec[i]
        D_total += (np.sum((ref_m - ctrl_m) ** 2) - C_ref[i] - ref.control_jackknife) / G_vec[i]
    chat, cref = float((C_hat / G_vec).sum()), float((C_ref / G_vec).sum())
    return {
        "rho": rho,
        "spread_B": spread_B,
        "credit": cr,
        "dist": float(dist),
        "chat": chat,
        "cref": cref,
        "credit_over_dist": cr / dist if dist > 0 else float("nan"),
        "chat_over_cref": chat / cref if cref > 0 else float("nan"),
        "num": float(N_total),
        "den": float(D_total),
        "expr_mse": float(N_total / D_total) if D_total >= 1e-12 else 1.0,
    }


def _not_own(G: int, target_idx: int) -> np.ndarray:
    own = np.ones(G, dtype=bool)
    if 0 <= target_idx < G:
        own[target_idx] = False
    return own


def direction_fidelity(pred_de: dict, ref_de: dict, target_idx: int) -> float:
    """fid: F = k / max(n_pred, n_real); adjudicable = reference lfc defined and non-zero; both silent -> NaN."""
    own = _not_own(len(pred_de["significant"]), target_idx)
    n_real = int((ref_de["significant"] & own).sum())
    rl = ref_de["lfc"]
    call = pred_de["significant"] & np.isfinite(rl) & (rl != 0) & own
    n_pred = int(call.sum())
    if n_real == 0 and n_pred == 0:
        return float("nan")
    if n_pred == 0:
        return 0.0
    ps = np.sign(pred_de["lfc"][call])
    k = int(np.sum((ps == np.sign(rl[call])) & (ps != 0) & np.isfinite(ps)))
    return float(k / max(n_pred, n_real))


def direction_reach(pred_de: dict, ref_de: dict, target_idx: int, genes: Sequence[str]) -> float:
    """reach: R = k*/n_real, the deepest prefix of the prediction's ranking of the reference budget with purity
    >= 0.9."""
    own = _not_own(len(pred_de["significant"]), target_idx)
    ref_sig = ref_de["significant"] & own
    budget = np.flatnonzero(ref_sig)
    n_real = int(budget.size)
    if n_real == 0:
        return float("nan")
    rl = ref_de["lfc"]
    adjudicable = ref_sig & np.isfinite(rl) & (rl != 0)
    padj, pval, lfc, sig = pred_de["padj"], pred_de["pval"], pred_de["lfc"], pred_de["significant"]

    def key(g):
        pj = padj[g] if np.isfinite(padj[g]) else np.inf
        pv = pval[g] if np.isfinite(pval[g]) else np.inf
        lf = abs(lfc[g]) if np.isfinite(lfc[g]) else 0.0
        return (not sig[g], pj, pv, -lf, str(genes[g]))

    correct = counted = k_star = 0
    for g in sorted(budget.tolist(), key=key):
        if not adjudicable[g]:
            continue
        counted += 1
        ps = np.sign(lfc[g]) if np.isfinite(lfc[g]) else 0.0
        if ps == np.sign(rl[g]) and ps != 0:
            correct += 1
        if correct / counted >= PURITY_THRESHOLD:
            k_star = counted
    return float(k_star / n_real)


def sig_jaccard(pred_de: dict, ref_de: dict, target_idx: int) -> float:
    """jac: Jaccard of the significant sets, own target removed; empty union = 1."""
    own = _not_own(len(pred_de["significant"]), target_idx)
    R = set(np.flatnonzero(ref_de["significant"] & own).tolist())
    Rh = set(np.flatnonzero(pred_de["significant"] & own).tolist())
    union = R | Rh
    return 1.0 if not union else len(R & Rh) / len(union)


def lfc_nmae(pred_de: dict, ref_de: dict, target_idx: int) -> float:
    """nmae: NMAE over the reference gate (significant, finite lfc, >= 10 genes); missing predicted lfc reads 0."""
    own = _not_own(len(ref_de["significant"]), target_idx)
    gate_idx = np.flatnonzero(ref_de["significant"] & np.isfinite(ref_de["lfc"]) & own)
    if gate_idx.size < NMAE_MIN_GATE:
        return float("nan")
    rl = ref_de["lfc"][gate_idx]
    pl = pred_de["lfc"][gate_idx]
    pl = np.where(np.isfinite(pl), pl, 0.0)
    den = float(np.abs(rl).sum())
    if den < 1e-12:
        return float("nan")
    return float(np.abs(rl - pl).sum() / den)


def score_panel(pred: SideStats, ref: SideStats, panel_targets) -> Dict[str, float]:
    """The six raw metrics plus the fid / nmae cohort sizes (undefined perturbations drop from the mean)."""
    fid, reach, jac, nmae = [], [], [], []
    for p in _shared(pred, ref):
        pd_de, rf_de = pred.de.get(p), ref.de.get(p)
        if pd_de is None or rf_de is None:
            continue
        t = ref.target_index(p)
        f = direction_fidelity(pd_de, rf_de, t)
        if np.isfinite(f):
            fid.append(f)
        r = direction_reach(pd_de, rf_de, t, pred.genes)
        if np.isfinite(r):
            reach.append(r)
        jac.append(sig_jaccard(pd_de, rf_de, t))
        nm = lfc_nmae(pd_de, rf_de, t)
        if np.isfinite(nm):
            nmae.append(nm)
    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    parts = expr_mse_parts(pred, ref)
    return {
        "pds": pds_cosine_panel(pred, ref, panel_targets),
        "expr_mse": parts.get("expr_mse", float("nan")),
        "fid": mean(fid),
        "reach": mean(reach),
        "jac": mean(jac),
        "nmae": mean(nmae),
        "n_fid": len(fid),
        "n_nmae": len(nmae),
    }


def scaled_scores(u: Dict[str, float], b: Dict[str, float], r: Dict[str, float]) -> Dict[str, float]:
    """s = (u - b)/(r - b); expr_mse clamped to [0, 1], nmae floored at -6, others unclamped; overall = mean."""
    out: Dict[str, float] = {}
    for m in METRIC_NAMES:
        span = r[m] - b[m]
        if abs(span) < 1e-12:
            out[m] = float("nan")
            continue
        s = (u[m] - b[m]) / span
        if m == "expr_mse":
            s = float(np.clip(s, 0.0, 1.0))
        elif m == "nmae":
            s = float(max(s, -6.0))
        out[m] = float(s)
    vals = [out[m] for m in METRIC_NAMES if np.isfinite(out[m])]
    out["overall"] = float(np.mean(vals)) if vals else float("nan")
    return out
