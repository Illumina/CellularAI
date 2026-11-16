"""Create a completely fictional summary catalog and prediction task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

GENES = [f"gene_{index:02d}" for index in range(30)]
PERTURBATIONS = ["gene_01", "gene_03", "gene_05"]
SOURCE_TARGETS = [*PERTURBATIONS, "gene_07", "gene_09", "target_extra"]
CELL_LINES = ["cell_line_A", "cell_line_B", "cell_line_C"]
AUXILIARY_LINE = "cell_line_auxiliary"


def _write_statistic(path: Path, values: np.ndarray) -> None:
    frame = pd.DataFrame(values, columns=[f"target__{target}" for target in SOURCE_TARGETS]).astype(np.float32)
    frame.insert(0, "feature_id", GENES)
    frame.to_parquet(path, index=False)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_world(output_dir: Path, source_config: Path) -> dict[str, Path]:
    """Write model configuration, catalog payloads, centering tables, variance data and controls."""
    output_dir = Path(output_dir).resolve()
    catalog_dir = output_dir / "customer_bundle"
    task_dir = output_dir / "task"
    summaries = catalog_dir / "summaries"
    offsets = catalog_dir / "offsets"
    for directory in (summaries, offsets, task_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20261113)
    rows = []
    variance_arrays = {"cell_lines": np.asarray(CELL_LINES)}
    for view in ("view_0", "view_1"):
        for line_index, line in enumerate(CELL_LINES):
            effect = rng.normal(0.0, 0.35, (len(GENES), len(SOURCE_TARGETS)))
            for column, target in enumerate(SOURCE_TARGETS):
                if target in GENES:
                    effect[GENES.index(target), column] = -0.7 - 0.1 * line_index
            probability = np.clip(rng.uniform(1e-10, 0.2, effect.shape) ** 2, 1e-12, 0.8)
            scale = rng.uniform(0.04, 0.4, effect.shape)
            controls = rng.integers(20, 2000, len(GENES)).astype(np.float64) * (1.0 + 0.15 * line_index)
            stem = f"{view}__{line}"
            paths = {
                "effect": summaries / f"{stem}__effect.parquet",
                "probability": summaries / f"{stem}__probability.parquet",
                "scale": summaries / f"{stem}__scale.parquet",
                "controls": summaries / f"{stem}__controls.parquet",
            }
            _write_statistic(paths["effect"], effect)
            _write_statistic(paths["probability"], probability)
            _write_statistic(paths["scale"], scale)
            pd.DataFrame({"feature_id": GENES, "control_sum": controls}).to_parquet(paths["controls"], index=False)
            offset = offsets / f"{stem}__offset.npz"
            np.savez(
                offset,
                cell_line=line,
                view=view,
                genes=np.asarray(GENES),
                effect_center=np.zeros(len(GENES)),
                effect_count=np.full(len(GENES), 100.0),
                signed_score_center=np.zeros(len(GENES)),
                response_variance=rng.uniform(0.02, 0.2, len(GENES)),
            )
            if view == "view_0":
                variance_arrays[f"genes__{line}"] = np.asarray(GENES)
                variance_arrays[f"variance__{line}"] = rng.uniform(0.02, 0.2, len(GENES))
            rows.append(
                {
                    "cell_line": line,
                    "view": view,
                    "role": "primary",
                    "files": {key: str(path.relative_to(catalog_dir)) for key, path in paths.items()},
                    "offset": str(offset.relative_to(catalog_dir)),
                    "pair_group": "paired_record" if line in CELL_LINES[:2] else None,
                    "pair_alternate": line == "cell_line_B",
                }
            )

    line = AUXILIARY_LINE
    view = "view_auxiliary"
    effect = rng.normal(0.0, 0.25, (len(GENES), len(SOURCE_TARGETS)))
    probability = rng.uniform(1e-8, 0.5, effect.shape)
    controls = rng.integers(20, 2000, len(GENES)).astype(np.float64)
    paths = {
        "effect": summaries / f"{view}__{line}__effect.parquet",
        "probability": summaries / f"{view}__{line}__probability.parquet",
        "controls": summaries / f"{view}__{line}__controls.parquet",
    }
    _write_statistic(paths["effect"], effect)
    _write_statistic(paths["probability"], probability)
    pd.DataFrame({"feature_id": GENES, "control_sum": controls}).to_parquet(paths["controls"], index=False)
    offset = offsets / f"{view}__{line}__offset.npz"
    np.savez(
        offset,
        cell_line=line,
        view=view,
        genes=np.asarray(GENES),
        effect_center=np.zeros(len(GENES)),
        effect_count=np.full(len(GENES), 50.0),
        signed_score_center=np.zeros(len(GENES)),
        response_variance=np.full(len(GENES), 0.1),
    )
    rows.append(
        {
            "cell_line": line,
            "view": view,
            "role": "auxiliary",
            "files": {key: str(path.relative_to(catalog_dir)) for key, path in paths.items()},
            "offset": str(offset.relative_to(catalog_dir)),
            "pair_group": None,
        }
    )

    variance = catalog_dir / "response_variance.npz"
    np.savez(variance, **variance_arrays)
    catalog_path = catalog_dir / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "primary_view": "view_0",
                "variance": variance.name,
                "orders": {
                    "effect_files": CELL_LINES,
                    "control_files": CELL_LINES,
                    "residual": CELL_LINES,
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    files = sorted(path for path in catalog_dir.rglob("*") if path.is_file())
    (catalog_dir / "SHA256SUMS").write_text(
        "".join(f"{_checksum(path)}  {path.relative_to(catalog_dir)}\n" for path in files)
    )

    control_rng = np.random.default_rng(81)
    rates = np.r_[control_rng.uniform(3.0, 35.0, len(GENES) - 1), 0.002]
    controls = control_rng.poisson(rates, (450, len(GENES))).astype(np.float32)
    for context in ("context_A", "context_B"):
        ad.AnnData(
            X=sp.csr_matrix(controls),
            var=pd.DataFrame(index=GENES),
        ).write_h5ad(task_dir / f"{context}_controls.h5ad")
    pd.DataFrame({"gene_name": GENES}).to_csv(task_dir / "readout.csv", index=False)
    pd.DataFrame({"target_gene": PERTURBATIONS[::-1]}).to_csv(task_dir / "targets.csv", index=False)
    task_path = output_dir / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "data_root": str(task_dir),
                "contexts": {
                    "context_A": {"controls": "context_A_controls.h5ad"},
                    "context_B": {"controls": "context_B_controls.h5ad"},
                },
                "perturbations": "targets.csv",
                "readout_genes": "readout.csv",
                "call_depth": 30,
                "median_degs": 8,
                "exclude_cell_lines": [],
                "control_label": "negative_control",
                "extra": {},
            },
            sort_keys=False,
        )
    )

    config_path = output_dir / "config.yaml"
    config = yaml.safe_load(Path(source_config).read_text())
    config["training_data"]["catalog"] = "customer_bundle/catalog.json"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return {
        "root": output_dir,
        "catalog": catalog_path,
        "task": task_path,
        "config": config_path,
        "controls": task_dir / "context_A_controls.h5ad",
    }
