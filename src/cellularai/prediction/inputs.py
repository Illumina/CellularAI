"""Load the training-summary matrices required for cell prediction."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import ndtri
from scipy.stats import norm

from ..pooling.catalog import Catalog, control_profiles, load_frame, primary_effect_files


@dataclass
class GenerationInputs:
    common: np.ndarray
    breadth_lines: list
    breadth_rel: np.ndarray
    reference_ctrl: dict
    mix_lines: list
    mix_z: np.ndarray
    sim_lines: list
    sim_z: np.ndarray
    sim_meas: np.ndarray
    residual_stack: np.ndarray
    residual_counts: np.ndarray
    residual_perts: list

    @classmethod
    def load(
        cls, training_data: dict, genes, perts, n_residual_perts: int, exclude=(), threads: int = 8
    ) -> "GenerationInputs":
        genes, perts = [str(gene) for gene in genes], [str(target) for target in perts]
        reference_ctrl = control_profiles(training_data, exclude)
        lines, relative_breadth = breadth_lines(training_data, perts, exclude)
        mix_lines = [line for line, _ in primary_effect_files(training_data, exclude)]
        sim_lines = sorted(reference_ctrl)
        with ThreadPoolExecutor(threads) as executor:
            mix = list(executor.map(lambda line: mix_line_z(training_data, line, perts, genes), mix_lines))
            sim = list(executor.map(lambda line: sim_line_z(training_data, line, genes, perts), sim_lines))
        stack, counts, residual_perts = residual_summaries(training_data, genes, n_residual_perts, exclude)
        return cls(
            common=common_sign(training_data, genes, perts, exclude),
            breadth_lines=lines,
            breadth_rel=relative_breadth,
            reference_ctrl=reference_ctrl,
            mix_lines=mix_lines,
            mix_z=np.stack(mix),
            sim_lines=sim_lines,
            sim_z=np.stack([values for values, _ in sim]),
            sim_meas=np.stack([measured for _, measured in sim]),
            residual_stack=stack,
            residual_counts=counts,
            residual_perts=residual_perts,
        )


def effect_lines(training_data, exclude) -> list[str]:
    return sorted({line for line, _ in primary_effect_files(training_data)} - set(exclude))


def common_sign(training_data, genes, perts, exclude) -> np.ndarray:
    gene_positions = {gene: i for i, gene in enumerate(genes)}
    accumulator = np.zeros(len(genes))
    catalog = Catalog.load(training_data)
    for cell_line in effect_lines(training_data, exclude):
        frame = load_frame(str(catalog.primary_row(cell_line).path("effect")), perts)
        rows = np.array([gene_positions.get(gene, -1) for gene in frame.index])
        effects = frame.to_numpy()[rows >= 0]
        accumulator[rows[rows >= 0]] += (effects > 0).sum(1) - (effects < 0).sum(1)
    common = np.sign(accumulator)
    common[common == 0] = 1.0
    return common


def breadth_lines(training_data, perts, exclude, chunk=400):
    """Calculate relative response breadth using every probability column, not only task targets."""
    wanted = {f"target__{target}": i for i, target in enumerate(perts)}
    lines = effect_lines(training_data, exclude)
    catalog = Catalog.load(training_data)
    relative = np.full((len(lines), len(perts)), np.nan)
    for line_index, cell_line in enumerate(lines):
        parquet = pq.ParquetFile(catalog.primary_row(cell_line).path("probability"))
        columns = [name for name in parquet.schema_arrow.names if name.startswith("target__")]
        breadth = np.full(len(columns), np.nan)
        for start in range(0, len(columns), chunk):
            table = parquet.read(columns=columns[start : start + chunk])
            matrix = np.stack(
                [
                    np.asarray(table.column(name).to_numpy(zero_copy_only=False), np.float64)
                    for name in table.column_names
                ],
                axis=1,
            )
            measured = np.isfinite(matrix).sum(0)
            with np.errstate(invalid="ignore"):
                significant = np.nansum(matrix < 1e-3, axis=0)
            breadth[start : start + chunk] = np.where(
                measured > 0,
                np.log(1e-5 + significant / np.maximum(measured, 1)),
                np.nan,
            )
        median = np.nanmedian(breadth)
        for column, value in zip(columns, breadth):
            target_index = wanted.get(column)
            if target_index is not None and np.isfinite(value):
                relative[line_index, target_index] = value - median
    return lines, relative


def mix_line_z(training_data, cell_line, perts, genes) -> np.ndarray:
    row = Catalog.load(training_data).primary_row(cell_line)
    effect = load_frame(str(row.path("effect")), perts, genes)
    probability = load_frame(str(row.path("probability")), perts, genes)
    effect_filled = np.where(np.isfinite(effect), effect, 0.0)
    probability_filled = np.clip(np.where(np.isfinite(probability), probability, 1.0), 1e-300, 1.0)
    values = -ndtri(probability_filled / 2.0) * np.sign(effect_filled)
    return np.where(np.isfinite(values), values, 0.0)


def _read_columns(path, columns):
    table = pq.read_table(path, columns=["feature_id"] + columns)
    genes = np.asarray(table.column("feature_id").to_pylist()).astype(str)
    matrix = (
        np.stack(
            [np.asarray(table.column(column).to_numpy(zero_copy_only=False), dtype=np.float64) for column in columns],
            axis=1,
        )
        if columns
        else np.zeros((len(genes), 0))
    )
    return genes, matrix


def sim_line_z(training_data, cell_line, genes, perts):
    """Return signed scores and measurement masks; duplicate genes retain their last row."""
    row = Catalog.load(training_data).primary_row(cell_line)
    effect_path, probability_path = row.path("effect"), row.path("probability")
    names = set(pq.ParquetFile(effect_path).schema_arrow.names) & set(
        pq.ParquetFile(probability_path).schema_arrow.names
    )
    target_columns = [f"target__{target}" for target in perts]
    present = [column for column in target_columns if column in names]
    gene_positions = {gene: i for i, gene in enumerate(genes)}
    column_indices = np.array([target_columns.index(column) for column in present], dtype=np.int64)

    def to_axis(row_genes, matrix):
        aligned = np.full((len(genes), len(perts)), np.nan)
        row_indices = np.array([gene_positions.get(gene, -1) for gene in row_genes])
        ok = row_indices >= 0
        if present:
            aligned[np.ix_(row_indices[ok], column_indices)] = matrix[ok]
        return aligned

    effect = to_axis(*_read_columns(effect_path, present))
    probability = to_axis(*_read_columns(probability_path, present))
    measured = np.isfinite(effect) & np.isfinite(probability)
    values = norm.isf(np.clip(np.where(measured, probability, 1.0), 1e-300, 1.0) / 2.0) * np.sign(
        np.where(measured, effect, 0.0)
    )
    return np.where(measured & np.isfinite(values), values, 0.0), measured


def residual_summaries(training_data, genes, n_perts: int, exclude):
    """Load deterministically selected residual targets in the recorded residual row order."""
    catalog = Catalog.load(training_data)
    excluded = set(exclude)
    lines = [line for line in catalog.orders["residual"] if line not in excluded]
    files = [(line, catalog.primary_row(line).path("effect")) for line in lines]
    pool = set()
    for _, path in files:
        pool.update(name.removeprefix("target__") for name in pq.read_schema(path).names if name.startswith("target__"))
    perts = sorted(sorted(pool, key=lambda name: (hashlib.sha256(name.encode()).digest(), name))[:n_perts])
    columns = ["target__" + target for target in perts]
    stack = np.full((len(files), len(genes), len(perts)), np.nan)
    counts = np.full((len(files), len(genes)), np.nan, dtype=np.float64)
    for index, (cell_line, path) in enumerate(files):
        names = set(pq.read_schema(path).names)
        table = pq.read_table(path, columns=["feature_id", *[column for column in columns if column in names]])
        table = table.to_pandas().set_index("feature_id")
        stack[index] = table.groupby(level=0).max().reindex(columns=columns).reindex(genes).to_numpy(dtype=np.float64)
        control_path = catalog.primary_row(cell_line).path("controls")
        if control_path is not None:
            controls = pq.read_table(control_path, columns=["feature_id", "control_sum"])
            raw = controls.column("control_sum").to_numpy().astype(np.float64)
            series = pd.Series(raw, index=controls.column("feature_id").to_pylist()).groupby(level=0).sum(min_count=1)
            counts[index] = series.reindex(genes).to_numpy(dtype=np.float64)
    return stack, counts, perts
