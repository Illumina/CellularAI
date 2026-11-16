"""Pool training summary statistics into context-specific effect estimates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import guards
from . import catalog as data
from .meta_analysis import CELL_LINE, combine_layer, parse_op
from .weights import Features, cell_line_factors


@dataclass
class Evidence:
    """Pooled effect and significance matrices on the task gene/perturbation axes."""

    l2: np.ndarray
    z: np.ndarray
    l2_tot: np.ndarray
    z_tot: np.ndarray
    guards: dict
    unmeasured: list

    def save(self, path, genes, perts) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            l2=self.l2,
            z=self.z,
            l2_tot=self.l2_tot,
            z_tot=self.z_tot,
            guard_names=np.asarray(list(self.guards), str),
            guard_counts=np.asarray(list(self.guards.values()), np.float64).reshape(-1, 3),
            unmeasured=np.asarray(self.unmeasured, str),
            genes=np.asarray(genes),
            perts=np.asarray(perts),
        )

    @classmethod
    def load(cls, path, genes, perts) -> "Evidence":
        with np.load(path, allow_pickle=False) as saved:
            if list(saved["genes"].astype(str)) != list(genes) or list(saved["perts"].astype(str)) != list(perts):
                raise ValueError(f"{path}: gene or perturbation axes differ from the task")
            return cls(
                *(saved[key] for key in ("l2", "z", "l2_tot", "z_tot")),
                {
                    str(name): [float(value) for value in counts]
                    for name, counts in zip(saved["guard_names"], saved["guard_counts"])
                },
                [str(target) for target in saved["unmeasured"]],
            )


def training_rows(catalog: data.Catalog, knobs, exclude=()) -> list[data.Row]:
    """Select configured primary views, then auxiliary rows, preserving catalog order within each."""
    excluded = set(exclude)
    primary = [row for view in knobs.views for row in catalog.by_view(view) if row.cell_line not in excluded]
    auxiliary = [row for row in catalog.rows if row.role == "auxiliary" and row.cell_line not in excluded]
    return primary + auxiliary


def build(ref: dict, knobs, limits, genes: list[str], perts: list[str], mean_cpm: np.ndarray, exclude=()) -> Evidence:
    start = time.monotonic()
    guards.reset()
    catalog = data.Catalog.load(ref)
    rows = training_rows(catalog, knobs, exclude)
    tables = data.load_tables(ref, catalog, rows)
    operations = knobs.ops()
    factor_names = {
        name for key, value in operations.items() for name in parse_op(value, key.rsplit("_", 1)[1])
    } & CELL_LINE
    factors = None
    if factor_names:
        features = Features(ref, perts, genes, mean_cpm, exclude)
        factors = cell_line_factors(features, knobs, factor_names, [row.cell_line for row in rows], genes, limits)
        del features
    if factors is not None and "sim" in factors.mat:
        guards.share(
            *zip(
                *[
                    (row.cell_line, factors.mat["sim"][i])
                    for row, i in zip(rows, factors.index)
                    if i is not None and row.role != "auxiliary"
                ]
            ),
            "pooling",
            limits,
        )
    else:
        guards.over(np.empty(0), limits.sim_share_max["pooling"], "sim_share_pooling")
        guards.under(np.empty(0), limits.sim_neff_min["pooling"], "sim_neff_pooling")
    print(
        f"[pooling] {len(rows)} summary rows, factors {sorted(factor_names)}: {time.monotonic() - start:.1f}s",
        flush=True,
    )
    effect_rows = [data.panel(row, "effect", genes, perts) for row in rows]
    probability_rows = [data.panel(row, "probability", genes, perts) for row in rows]
    multipliers = [knobs.auxiliary_weight if row.role == "auxiliary" else 1.0 for row in rows]
    unmeasured = [
        target for i, target in enumerate(perts) if not any(np.isfinite(matrix[:, i]).any() for matrix in effect_rows)
    ]
    l2, z = combine_layer(
        effect_rows, probability_rows, tables, genes, operations, multipliers, limits.v_floor_rel, factors
    )
    l2 = guards.clip_abs(l2, limits.l2_abs_max, "l2_abs_max")
    z = guards.clip_abs(z, limits.z_abs_max, "z_abs_max")
    tally = guards.snapshot()
    print(f"[pooling] complete in {time.monotonic() - start:.1f}s; guards {tally}; unmeasured {unmeasured}", flush=True)
    return Evidence(l2, z, l2.copy(), z.copy(), tally, unmeasured)
