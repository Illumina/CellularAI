#!/usr/bin/env python3
"""Build and validate a tiny synthetic prediction without network access or external data.

Run from the public source checkout with ``PYTHONPATH=src python examples/synthetic/run_example.py
--output-dir OUTPUT``. The script copies ``configs/example.yaml``, changes only its catalog path,
runs pooling and prediction for ``context_A``, and checks the resulting cell
matrix, feature labels, target labels and equal-sized (default 400-cell) target blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from synthetic_data import PERTURBATIONS, create_world

from cellularai.pooling.__main__ import main as pool
from cellularai.prediction.__main__ import main as predict
from cellularai.task import NCELL


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source_config = Path(__file__).resolve().parents[2] / "configs" / "example.yaml"
    world = create_world(args.output_dir, source_config)

    context = "context_A"
    build_dir = args.output_dir.resolve() / "builds" / f"ctx_{context}"
    evidence_path = build_dir / f"pooled_effects_{context}.npz"
    prediction_path = build_dir / f"predicted_cells_{context}.h5ad"
    pool(
        [
            "--config",
            str(world["config"]),
            "--task",
            str(world["task"]),
            "--context",
            context,
            "--out",
            str(evidence_path),
        ]
    )
    predict(
        [
            "--config",
            str(world["config"]),
            "--task",
            str(world["task"]),
            "--context",
            context,
            "--pooling",
            str(evidence_path),
            "--out",
            str(prediction_path),
            "--n-jobs",
            "1",
        ]
    )

    prediction = ad.read_h5ad(prediction_path)
    targets = prediction.obs["target_gene"].astype(str).tolist()
    expected_targets = [target for target in sorted(PERTURBATIONS) for _ in range(NCELL)]
    assert prediction.shape == (len(PERTURBATIONS) * NCELL, 30), prediction.shape
    assert list(prediction.var_names) == [f"gene_{index:02d}" for index in range(30)]
    assert targets == expected_targets
    assert prediction.obs["context"].astype(str).unique().tolist() == [context]
    assert sp.issparse(prediction.X) and prediction.X.dtype == np.float32
    values = prediction.X.toarray()
    assert np.isfinite(values).all() and (values >= 0).all()
    assert np.array_equal(values, np.rint(values))
    counts = prediction.obs["target_gene"].value_counts()
    assert (counts.reindex(sorted(PERTURBATIONS)) == NCELL).all()
    print(f"Synthetic example passed: {prediction.shape[0]} cells, {prediction.shape[1]} genes, 3 targets.")
    print(f"Prediction: {prediction_path}")


if __name__ == "__main__":
    main()
