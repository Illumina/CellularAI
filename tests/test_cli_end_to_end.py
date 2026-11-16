"""Synthetic end-to-end pooling and prediction CLI tests."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from conftest import GENES, PERTURBATIONS, SOURCE_CONFIG, create_world

from cellularai import assemble
from cellularai.config import Config
from cellularai.guards import NAMES
from cellularai.io import code_version
from cellularai.pooling.__main__ import main as pool_main
from cellularai.pooling.build import Evidence
from cellularai.prediction.__main__ import main as predict_main
from cellularai.task import NCELL, Task


def prediction_args(world, evidence, output):
    return [
        "--config",
        str(world["config"]),
        "--task",
        str(world["task"]),
        "--context",
        "context_A",
        "--pooling",
        str(evidence),
        "--out",
        str(output),
        "--n-jobs",
        "1",
    ]


@pytest.fixture(scope="module")
def module_world(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli_world")
    world = create_world(root, SOURCE_CONFIG)
    evidence = root / "builds" / "pooled.npz"
    prediction = root / "builds" / "prediction.h5ad"
    pool_main(
        [
            "--config",
            str(world["config"]),
            "--task",
            str(world["task"]),
            "--context",
            "context_A",
            "--out",
            str(evidence),
        ]
    )
    predict_main(prediction_args(world, evidence, prediction))
    return world, evidence, prediction


def test_pooling_cli_writes_finite_matrices_on_sorted_task_axes(module_world):
    world, evidence_path, _ = module_world
    config = Config.load(world["config"])
    task = Task.load(world["task"], config.training_data)
    evidence = Evidence.load(evidence_path, task.genes, task.perts)
    assert task.perts == sorted(PERTURBATIONS)
    assert evidence.l2.shape == evidence.z.shape == (len(GENES), len(PERTURBATIONS))
    assert np.isfinite(evidence.l2).all() and np.isfinite(evidence.z).all()
    np.testing.assert_array_equal(evidence.l2, evidence.l2_tot)
    np.testing.assert_array_equal(evidence.z, evidence.z_tot)
    assert evidence.unmeasured == [] and (evidence.z >= 0).all()
    assert np.abs(evidence.l2).max() > 0


def test_prediction_cli_writes_integer_cells_and_complete_provenance(module_world):
    world, _, prediction_path = module_world
    prediction = ad.read_h5ad(prediction_path)
    assert prediction.shape == (NCELL * len(PERTURBATIONS), len(GENES))
    assert sp.issparse(prediction.X) and prediction.X.dtype == np.float32
    assert list(prediction.var_names) == GENES
    assert prediction.obs["target_gene"].tolist() == [target for target in sorted(PERTURBATIONS) for _ in range(NCELL)]
    assert prediction.obs["context"].astype(str).unique().tolist() == ["context_A"]
    assert prediction.obs_names[0] == "context_A_gene_01_0"
    values = prediction.X.toarray()
    assert np.isfinite(values).all() and values.min() >= 0
    assert np.array_equal(values, np.rint(values)) and (values.sum(axis=1) > 0).all()
    assert prediction.obs["target_gene"].value_counts().eq(NCELL).all()
    report = prediction.uns["build"]
    assert set(report["guards"]) == set(NAMES)
    assert list(report["readout"]["missing"]) == []
    assert report["provenance"]["package_version"] == code_version()["package_version"]
    assert report["provenance"]["code_sha256"] == code_version()["code_sha256"]


def test_prediction_cli_rerun_is_bit_identical(module_world, tmp_path):
    world, evidence, _ = module_world
    first = ad.read_h5ad(world["root"] / "builds" / "prediction.h5ad")
    output = tmp_path / "rerun.h5ad"
    predict_main(prediction_args(world, evidence, output))
    second = ad.read_h5ad(output)
    for key in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(first.X, key), getattr(second.X, key))
    assert first.obs.equals(second.obs) and first.var.equals(second.var)


def test_target_file_order_does_not_change_generated_counts(module_world, tmp_path):
    world, evidence, _ = module_world
    first = ad.read_h5ad(world["root"] / "builds" / "prediction.h5ad")
    targets = world["task"].parent / "task" / "targets.csv"
    pd.DataFrame({"target_gene": list(reversed(PERTURBATIONS))}).to_csv(targets, index=False)
    output = tmp_path / "reordered.h5ad"
    predict_main(prediction_args(world, evidence, output))
    second = ad.read_h5ad(output)
    assert first.obs["target_gene"].tolist() == second.obs["target_gene"].tolist()
    for key in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(first.X, key), getattr(second.X, key))


def test_cells_per_perturbation_sets_the_generated_block_size(tmp_path):
    world = create_world(tmp_path, SOURCE_CONFIG)
    payload = yaml.safe_load(world["task"].read_text())
    world["task"].write_text(yaml.safe_dump(payload | {"cells_per_perturbation": 50}, sort_keys=False))
    evidence, output = tmp_path / "pooled.npz", tmp_path / "prediction.h5ad"
    pool_main(
        ["--config", str(world["config"]), "--task", str(world["task"]), "--context", "context_A", "--out", str(evidence)]
    )
    predict_main(prediction_args(world, evidence, output))
    prediction = ad.read_h5ad(output)
    assert prediction.shape == (50 * len(PERTURBATIONS), len(GENES))
    assert prediction.obs["target_gene"].value_counts().eq(50).all()
    assert prediction.obs_names[-1] == f"context_A_{sorted(PERTURBATIONS)[-1]}_49"
    config = Config.load(world["config"])
    task = Task.load(world["task"], config.training_data)
    expected = {"config": config.fingerprint, "task": task.fingerprint}
    checks = assemble.gate(
        prediction, "context_A", task.genes, task.perts, task.controls("context_A"), 30, config.model.guards, expected,
        task.n_cells,
    )
    assert checks["cells_per_perturbation"]
    default_size = assemble.gate(
        prediction, "context_A", task.genes, task.perts, task.controls("context_A"), 30, config.model.guards, expected
    )
    assert not default_size["cells_per_perturbation"]


def test_task_seed_draws_reproducible_independent_replicates(module_world, tmp_path):
    world, evidence, _ = module_world
    released = ad.read_h5ad(world["root"] / "builds" / "prediction.h5ad").X.toarray()
    original = world["task"].read_text()
    payload = yaml.safe_load(original)
    try:
        replicates = []
        for seed in (1, 1, 2):
            world["task"].write_text(yaml.safe_dump(payload | {"seed": seed}, sort_keys=False))
            output = tmp_path / f"seed_{seed}_{len(replicates)}.h5ad"
            predict_main(prediction_args(world, evidence, output))
            replicates.append(ad.read_h5ad(output).X.toarray())
    finally:
        world["task"].write_text(original)
    np.testing.assert_array_equal(replicates[0], replicates[1])
    assert not np.array_equal(replicates[0], replicates[2])
    assert not np.array_equal(replicates[0], released)


def test_prediction_workers_default_to_compute_workers(module_world, tmp_path, monkeypatch):
    world, evidence, _ = module_world
    seen = []
    import cellularai.prediction.__main__ as prediction_main

    def fake_build(*args, n_jobs, **kwargs):
        seen.append(n_jobs)
        raise RuntimeError("stop")

    monkeypatch.setattr(prediction_main, "build", fake_build)
    args = prediction_args(world, evidence, tmp_path / "unused.h5ad")[:-2]
    with pytest.raises(RuntimeError, match="stop"):
        predict_main(args)
    assert seen == [int(yaml.safe_load(world["config"].read_text())["compute"].get("workers", 1))]
