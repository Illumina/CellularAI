"""Read the portable summary-data catalog used for prediction.

The catalog contains only summary statistics, control-expression profiles, centering estimates and variance estimates. Every path is relative to the catalog unless a user explicitly supplies an absolute path. Row orders are recorded explicitly because floating-point accumulation and several linear-algebra operations are order-sensitive.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VARIANCE_FLOOR = 1e-8


def _catalog_path(ref) -> Path:
    return Path(ref["catalog"] if isinstance(ref, dict) else ref)


@dataclass(frozen=True)
class Row:
    """One summary view for one training cell line."""

    cell_line: str
    view: str
    role: str
    files: dict[str, str] = field(default_factory=dict)
    offset: str | None = None
    pair_group: str | None = None
    pair_alternate: bool = False
    root: Path | None = None

    @property
    def key(self) -> str:
        return f"{self.view}/{self.cell_line}"

    def path(self, statistic: str) -> Path | None:
        value = self.files.get(statistic)
        if not value:
            return None
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{self.key}: summary payload paths must remain inside the bundle")
        return self.root / path

    def offset_path(self) -> Path:
        if not self.offset:
            raise ValueError(f"{self.key}: no centering table")
        path = Path(self.offset)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{self.key}: centering paths must remain inside the bundle")
        return self.root / path


@dataclass
class Catalog:
    rows: list[Row]
    root: Path
    primary_view: str
    variance: str
    orders: dict[str, list[str]]

    @classmethod
    def load(cls, ref) -> "Catalog":
        path = _catalog_path(ref).resolve()
        payload = json.loads(path.read_text())
        root = path.parent
        rows = [Row(**row, root=root) for row in payload["rows"]]
        return cls(
            rows=rows,
            root=root,
            primary_view=payload["primary_view"],
            variance=payload["variance"],
            orders={key: list(value) for key, value in payload["orders"].items()},
        )

    def by_view(self, view: str) -> list[Row]:
        return [row for row in self.rows if row.view == view]

    @property
    def cell_lines(self) -> set[str]:
        return {row.cell_line for row in self.rows}

    def primary_row(self, cell_line: str) -> Row:
        for row in self.rows:
            if row.cell_line == cell_line and row.view == self.primary_view and row.role == "primary":
                return row
        raise KeyError(f"no primary summary for {cell_line!r}")

    def variance_path(self) -> Path:
        path = Path(self.variance)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("the variance path must remain inside the bundle")
        return self.root / path


@dataclass
class OffsetTable:
    """Per-gene centering and response-variance estimates for one summary row."""

    cell_line: str
    view: str
    genes: np.ndarray
    effect_center: np.ndarray
    effect_count: np.ndarray
    signed_score_center: np.ndarray
    response_variance: np.ndarray

    @classmethod
    def load(cls, path) -> "OffsetTable":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                cell_line=str(data["cell_line"]),
                view=str(data["view"]),
                genes=data["genes"].astype(str),
                effect_center=data["effect_center"],
                effect_count=data["effect_count"],
                signed_score_center=data["signed_score_center"],
                response_variance=data["response_variance"],
            )


def load_tables(ref, catalog: Catalog, rows: list[Row]) -> list[OffsetTable]:
    tables = [OffsetTable.load(row.offset_path()) for row in rows]
    for row, table in zip(rows, tables):
        if (table.cell_line, table.view) != (row.cell_line, row.view):
            raise ValueError(f"{row.key}: centering table identifiers do not match the catalog")
    return tables


def signed_z(effect: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Convert two-sided probabilities to signed normal scores without upper-tail underflow."""
    from scipy.special import ndtri

    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = -ndtri(np.clip(probability, 1e-300, 1.0) / 2.0)
    out = magnitude * np.sign(effect)
    return np.where(np.isfinite(effect) & np.isfinite(probability), out, np.nan)


def apply_offsets(panel: np.ndarray, genes, table: OffsetTable, stream: str) -> np.ndarray:
    """Subtract finite centers; genes without an estimate pass through unchanged."""
    center = {"l2": table.effect_center, "z": table.signed_score_center}[stream]
    positions = {gene: i for i, gene in enumerate(table.genes)}
    idx = np.fromiter((positions.get(str(gene), -1) for gene in genes), dtype=np.int64, count=len(genes))
    offset = np.where(idx >= 0, center[np.maximum(idx, 0)], np.nan)
    offset = np.where(np.isfinite(offset), offset, 0.0)[:, None]
    return np.where(np.isfinite(panel), panel - offset, panel)


def primary_effect_files(ref, exclude=()) -> list[tuple[str, str]]:
    """Primary effect tables in the explicit generation/pooling order."""
    catalog = Catalog.load(ref)
    excluded = set(exclude)
    return [
        (cell_line, str(catalog.primary_row(cell_line).path("effect")))
        for cell_line in catalog.orders["effect_files"]
        if cell_line not in excluded
    ]


def check_exclude(ref: dict, exclude) -> None:
    """Fail before computation if an excluded cell line is absent from any required catalog axis."""
    catalog = Catalog.load(ref)
    primary = {line for line, _ in primary_effect_files(ref)}
    controls = set(catalog.orders["control_files"])
    known = {"catalog": catalog.cell_lines, "effects": primary, "control_profiles": controls}
    if missing := {key: sorted(set(exclude) - values) for key, values in known.items() if set(exclude) - values}:
        raise ValueError(f"exclude_cell_lines contains unknown training cell lines: {missing}")


def load_frame(path, perturbations: list[str], genes=None) -> pd.DataFrame | np.ndarray:
    """Read neutral effect-like columns, collapsing duplicate genes by maximum as in the model definition."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    rows = parquet.read(columns=["feature_id"]).to_pandas()["feature_id"].astype(str)
    wanted = [f"target__{target}" for target in perturbations if f"target__{target}" in parquet.schema_arrow.names]
    matrix = parquet.read(columns=wanted).to_pandas()
    matrix.index = rows.to_numpy()
    matrix = matrix.groupby(level=0).max()
    matrix.columns = [column.removeprefix("target__") for column in matrix.columns]
    matrix = matrix.reindex(columns=perturbations)
    return matrix if genes is None else matrix.reindex(index=list(genes)).to_numpy(dtype=np.float64)


def panel(row: Row, statistic: str, genes, perturbations) -> np.ndarray:
    path = row.path(statistic)
    if path is None:
        return np.full((len(genes), len(perturbations)), np.nan)
    return load_frame(str(path), list(perturbations), genes)


def name_key(name: str) -> str:
    return re.sub(r"\.\d+$", "", str(name).strip()).upper()


def readout_coverage(genes, rows: list[Row]) -> dict:
    """Report readouts absent from every effect axis and simple name-only near matches."""
    import pyarrow.parquet as pq

    axis = set()
    for row in rows:
        axis.update(pq.read_table(row.path("effect"), columns=["feature_id"]).column("feature_id").to_pylist())
    axis = {str(gene) for gene in axis}
    missing = [gene for gene in genes if gene not in axis]
    keys: dict[str, list[str]] = {}
    for gene in sorted(axis):
        keys.setdefault(name_key(gene), []).append(gene)
    near = [f"{gene} -> {candidate}" for gene in missing for candidate in keys.get(name_key(gene), [])]
    return {"n_genes": len(genes), "n_training_genes": len(axis), "missing": missing, "near_miss": near}


def log2fc_stack(perturbations: list[str], ref, exclude=()) -> tuple[np.ndarray, list[str], list[str]]:
    """Stack primary effects on their sorted gene union in the recorded file order."""
    lines, matrices = [], []
    for cell_line, path in primary_effect_files(ref, exclude):
        matrices.append(load_frame(path, perturbations))
        lines.append(cell_line)
    union = sorted(set().union(*(set(matrix.index) for matrix in matrices)))
    stack = np.full((len(matrices), len(union), len(perturbations)), np.nan)
    positions = {gene: i for i, gene in enumerate(union)}
    for i, matrix in enumerate(matrices):
        common = [gene for gene in union if gene in matrix.index]
        stack[i][[positions[gene] for gene in common]] = matrix.loc[common].to_numpy(dtype=np.float64)
    return stack, lines, union


def control_profiles(ref, exclude=()) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return control-expression sums in the explicitly recorded control-file order."""
    import pyarrow.parquet as pq

    catalog = Catalog.load(ref)
    excluded = set(exclude)
    out = {}
    for cell_line in catalog.orders["control_files"]:
        if cell_line in excluded:
            continue
        path = catalog.primary_row(cell_line).path("controls")
        table = pq.read_table(path, columns=["feature_id", "control_sum"])
        out[cell_line] = (
            np.array([str(gene) for gene in table.column("feature_id").to_pylist()]),
            table.column("control_sum").to_numpy().astype(np.float64),
        )
    return out


def inverse_variance_weights(stack: np.ndarray) -> np.ndarray:
    measured = np.isfinite(stack)
    filled = np.where(measured, stack, 0.0).astype(np.float64)
    var_filled = filled.var(axis=2)
    row = measured.any(axis=2)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gene_typ = np.nanmedian(np.where(row, var_filled, np.nan), axis=0)
    if np.isnan(gene_typ).any():
        finite = gene_typ[np.isfinite(gene_typ)]
        gene_typ = np.where(np.isfinite(gene_typ), gene_typ, float(np.median(finite)) if finite.size else 1.0)
    var_eff = np.where(row, var_filled, gene_typ[None, :])
    return 1.0 / np.maximum(var_eff, VARIANCE_FLOOR)


def load_null_variance(ref, lines, union) -> np.ndarray:
    """Align primary response-variance estimates to the stacked gene union."""
    path = Catalog.load(ref).variance_path()
    with np.load(path, allow_pickle=False) as data:
        have = set(data["cell_lines"].astype(str))
        if any(line not in have for line in lines):
            raise ValueError(f"{path}: missing variance records")
        positions = {gene: i for i, gene in enumerate(union)}
        variance = np.full((len(lines), len(union)), np.nan)
        for i, cell_line in enumerate(lines):
            idx = np.array([positions.get(gene, -1) for gene in data[f"genes__{cell_line}"].astype(str)])
            ok = idx >= 0
            variance[i, idx[ok]] = data[f"variance__{cell_line}"][ok]
    return variance
