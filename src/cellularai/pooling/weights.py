"""Compute training-cell-line weights from summary evidence and control profiles."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from .. import guards
from .catalog import (
    VARIANCE_FLOOR,
    Catalog,
    control_profiles,
    inverse_variance_weights,
    load_frame,
    load_null_variance,
    log2fc_stack,
)


def sim_weight(correlation: np.ndarray, beta: float, sd_min: float, z_clip: float) -> np.ndarray:
    correlation = np.asarray(correlation, np.float64)
    sd = guards.floor(np.nanstd(correlation), sd_min, "sim_sd_min")
    return np.exp(beta * guards.clip_abs((correlation - np.nanmean(correlation)) / sd, z_clip, "sim_z_clip"))


class Features:
    """Training features on the primary view's cell-line by gene-union axes."""

    def __init__(self, ref, perts, genes, target_cpm, exclude=()):
        self.stack, self.lines, self.union = log2fc_stack(list(perts), ref, exclude)
        self.ref, self.perts, self.genes = ref, list(perts), list(genes)
        union_positions = {gene: i for i, gene in enumerate(self.union)}
        self.cpm = np.full((len(self.lines), len(self.union)), np.nan)
        profiles = control_profiles(ref, exclude)
        for i, cell_line in enumerate(self.lines):
            line_genes, counts = profiles[cell_line]
            idx = np.array([union_positions.get(gene, -1) for gene in line_genes])
            ok = idx >= 0
            self.cpm[i, idx[ok]] = 1e6 * counts[ok] / counts.sum()
        panel_positions = {gene: i for i, gene in enumerate(self.genes)}
        target_cpm = np.asarray(target_cpm, np.float64)
        self.tcpm = np.array(
            [target_cpm[panel_positions[gene]] if gene in panel_positions else np.nan for gene in self.union]
        )
        target_indices = np.array([union_positions.get(target, -1) for target in self.perts])
        own_effect = np.where(
            target_indices[None] >= 0,
            self.stack[:, np.maximum(target_indices, 0), np.arange(len(self.perts))],
            np.nan,
        )
        self.knockdown = 1.0 - 2.0 ** own_effect.astype(np.float64)
        catalog = Catalog.load(ref)
        self.pair_groups = [catalog.primary_row(line).pair_group for line in self.lines]
        self.pair_alternates = [catalog.primary_row(line).pair_alternate for line in self.lines]

    def kd_factor(self, ref: float, floor: float) -> np.ndarray:
        return np.where(np.isfinite(self.knockdown), np.clip(self.knockdown / ref, floor, 1.0), 1.0)

    def sim_factor(self, beta: float, n_hvg: int, sd_min: float, z_clip: float) -> np.ndarray:
        log_profiles = np.log1p(self.cpm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            variance = np.nanvar(log_profiles, axis=0)
        highly_variable = np.argsort(np.nan_to_num(variance))[-int(n_hvg) :]
        target = np.log1p(self.tcpm[highly_variable])
        correlations = np.array(
            [spearmanr(log_profiles[i, highly_variable], target, nan_policy="omit")[0] for i in range(len(self.lines))]
        )
        return sim_weight(correlations, beta, sd_min, z_clip)

    def pair_factor(self, value: float) -> np.ndarray:
        """Down-weight pairs and alternate records, including an alternate without its base."""
        counts = {group: self.pair_groups.count(group) for group in set(self.pair_groups) if group is not None}
        return np.array(
            [
                value if alternate or (group is not None and counts[group] >= 2) else 1.0
                for group, alternate in zip(self.pair_groups, self.pair_alternates)
            ]
        )

    def noise_ratio(self, alpha: float, bound: float) -> np.ndarray:
        variance = load_null_variance(self.ref, self.lines, self.union)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gene_median = np.nanmedian(np.where(variance > 0, variance, np.nan), axis=0)
            gene_median = np.where(np.isfinite(gene_median), gene_median, np.nanmedian(gene_median))
        variance = np.where(np.isfinite(variance) & (variance > 0), variance, gene_median[None])
        ratio = (1.0 / np.maximum(variance, VARIANCE_FLOOR) / inverse_variance_weights(self.stack)) ** alpha
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        ratio = ratio / np.median(ratio)
        with np.errstate(divide="ignore"):
            guards.over(np.maximum(ratio, 1.0 / ratio), bound, "nr_clip")
        return np.clip(ratio, 1.0 / bound, bound)

    def se_factor(self, alpha: float, clip: float) -> np.ndarray:
        catalog = Catalog.load(self.ref)
        out = np.ones((len(self.lines), len(self.perts)))
        for i, cell_line in enumerate(self.lines):
            path = catalog.primary_row(cell_line).path("scale")
            if path is None or not path.exists():
                continue
            scale = load_frame(str(path), self.perts).to_numpy(np.float64)
            scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                medians = np.nanmedian(scale, axis=0)
                ratio = (np.nanmedian(medians) / medians) ** (2.0 * alpha)
                out[i] = np.where(np.isfinite(ratio), np.clip(ratio, 1.0 / clip, clip), 1.0)
        return out


@dataclass
class CellLineFactors:
    """Weight factors indexed against summary rows and panel genes."""

    index: list
    gidx: np.ndarray
    n_perts: int
    row: dict = field(default_factory=dict)
    col: dict = field(default_factory=dict)
    mat: dict = field(default_factory=dict)

    ORDER = {"row": ("nr",), "col": ("kd", "sef"), "mat": ("sim", "pair")}

    @classmethod
    def on_rows(cls, lines, union, row_lines, genes, n_perts, row=None, col=None, mat=None):
        positions = {line: i for i, line in enumerate(lines)}
        union_positions = {gene: i for i, gene in enumerate(union)}
        return cls(
            index=[positions.get(line) for line in row_lines],
            gidx=np.array([union_positions.get(str(gene), -1) for gene in genes], dtype=np.int64),
            n_perts=int(n_perts),
            row=dict(row or {}),
            col=dict(col or {}),
            mat=dict(mat or {}),
        )

    def block(self, index: int, names) -> np.ndarray | None:
        line_index = self.index[index]
        names = set(names)
        if line_index is None or not names:
            return None
        measured = self.gidx >= 0
        union_indices = self.gidx[measured]
        gene_weights = np.ones(union_indices.size)
        for name in self.ORDER["row"]:
            if name in names:
                gene_weights = gene_weights * self.row[name][line_index][union_indices]
        line_weight = np.float64(1.0)
        for name in self.ORDER["mat"]:
            if name in names:
                line_weight = line_weight * self.mat[name][line_index]
        gene_weights = gene_weights * line_weight
        target_weights = np.ones(self.n_perts)
        for name in self.ORDER["col"]:
            if name in names:
                target_weights = target_weights * self.col[name][line_index]
        out = np.full((self.gidx.size, self.n_perts), np.nan)
        out[measured] = gene_weights[:, None] * target_weights[None, :]
        return np.nan_to_num(out, nan=1.0)


def cell_line_factors(features: Features, knobs, names, row_lines, genes, limits) -> CellLineFactors:
    row, col, mat = {}, {}, {}
    if "nr" in names:
        row["nr"] = features.noise_ratio(knobs.noise, limits.nr_clip)
    if "sef" in names:
        col["sef"] = features.se_factor(knobs.se, knobs.se_clip)
    if "kd" in names:
        col["kd"] = features.kd_factor(knobs.kd_ref, knobs.kd_floor)
    if "sim" in names:
        mat["sim"] = features.sim_factor(knobs.sim, knobs.sim_hvg, limits.sim_sd_min, limits.sim_z_clip)
    if "pair" in names:
        mat["pair"] = features.pair_factor(knobs.pair_weight)
    return CellLineFactors.on_rows(features.lines, features.union, row_lines, genes, len(features.perts), row, col, mat)
