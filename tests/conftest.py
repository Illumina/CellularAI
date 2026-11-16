"""Shared synthetic worlds for the public tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from cellularai.evaluation.data import ANCHOR_CODE_VERSION, FILE_KEYS
from cellularai.evaluation.scoring import build_prediction_side, jackknife_moment
from cellularai.guards import NAMES
from cellularai.io import md5
from cellularai.task import NCELL, Task

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "example.yaml"
SYNTHETIC_DIR = ROOT / "examples" / "synthetic"
if str(SYNTHETIC_DIR) not in sys.path:
    sys.path.insert(0, str(SYNTHETIC_DIR))

from synthetic_data import CELL_LINES, GENES, PERTURBATIONS, create_world  # noqa: E402,F401


@pytest.fixture
def world(tmp_path):
    return create_world(tmp_path / "world", SOURCE_CONFIG)


@pytest.fixture
def model_payload():
    return yaml.safe_load(SOURCE_CONFIG.read_text())["model"]


def clean_prediction(world, path: Path, context: str = "context_A") -> ad.AnnData:
    from cellularai.config import Config

    config = Config.load(world["config"])
    task = Task.load(world["task"], config.training_data)
    controls = task.controls(context).toarray()
    rng = np.random.default_rng(19)
    perts = task.perts
    block = np.concatenate([controls[rng.choice(controls.shape[0], NCELL, replace=False)] for _ in perts]).astype(
        np.float32
    )
    observations = pd.DataFrame(
        {
            "context": [context] * len(block),
            "target_gene": [target for target in perts for _ in range(NCELL)],
        }
    )
    observations.index = [f"{context}_{target}_{index}" for target in perts for index in range(NCELL)]
    report = {
        "guards": {name: [0, 4_000_000 if name == "nr_clip" else 1_000_000, np.nan] for name in NAMES},
        "unmeasured": {"no_row": [], "no_evidence": []},
        "readout": {"n_genes": len(task.genes), "missing": [], "near_miss": []},
        "provenance": {"config": config.fingerprint, "task": task.fingerprint},
        "margin_reached": True,
        "converged": True,
        "credit_share": 0.99,
    }
    prediction = ad.AnnData(
        X=sp.csr_matrix(block),
        obs=observations,
        var=pd.DataFrame(index=task.genes),
        uns={"build": report},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(path)
    return prediction


def make_evaluation_bundle(root: Path, public_id: str) -> dict[str, Path]:
    """Create a small, fictional evaluation bundle for either supported public identifier."""
    root = root / public_id
    root.mkdir(parents=True)
    rng = np.random.default_rng(107)
    genes = [f"gene_{index:02d}" for index in range(48)]
    perts = ["gene_00", "gene_01", "gene_02"]
    pool = rng.poisson(20.0, (360, len(genes))).astype(np.float32)
    provision = np.sort(rng.choice(pool.shape[0], 180, replace=False))
    np.savez(root / "controls.npz", counts=pool, genes=np.asarray(genes))
    np.save(root / "provision.npy", provision)
    (root / "targets.json").write_text(json.dumps(perts))
    truth = {}
    for index, target in enumerate(perts):
        rates = np.full(len(genes), 20.0)
        start = 5 + 12 * index
        rates[start : start + 12] = 90.0
        truth[target] = rng.poisson(rates, (NCELL, len(genes))).astype(np.float32)
    controls = pool[provision]
    reference = build_prediction_side(truth, controls, genes, perts)
    reference.control_jackknife = jackknife_moment(controls)
    arrays = {
        "control_profile": reference.control_profile,
        "control_jackknife": np.float64(reference.control_jackknife),
        "perts": np.asarray(reference.perts),
        "genes": np.asarray(reference.genes),
        "jackknife_perts": np.asarray(list(reference.jackknife)),
        "jackknife_vals": np.asarray(list(reference.jackknife.values())),
    }
    for index, target in enumerate(reference.perts):
        arrays[f"profile_{index}"] = reference.profiles[target]
        for key, value in reference.de[target].items():
            arrays[f"de{index}_{key}"] = value
    np.savez_compressed(root / "reference.npz", **arrays)
    baseline = dict(pds=0.5, expr_mse=1.0, fid=0.1, reach=0.1, jac=0.1, nmae=1.0)
    replicate = dict(pds=1.0, expr_mse=0.0, fid=0.9, reach=0.9, jac=0.6, nmae=0.3)
    (root / "anchors.json").write_text(
        json.dumps(
            {
                "code_version": ANCHOR_CODE_VERSION,
                "baseline": baseline,
                "replicate": replicate,
            }
        )
    )
    names = dict(zip(FILE_KEYS, ("controls.npz", "targets.json", "provision.npy", "anchors.json", "reference.npz")))
    checksums = {key: md5(root / filename) for key, filename in names.items()}
    config_path = root / "evaluation.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"datasets": {public_id: {"root": str(root), "files": names, "md5": checksums}}}, sort_keys=False
        )
    )

    block = np.concatenate([truth[target] for target in perts])
    observations = pd.DataFrame({"target_gene": [target for target in perts for _ in range(NCELL)]})
    observations.index = observations.index.astype(str)
    prediction_path = root / "prediction.h5ad"
    ad.AnnData(
        X=sp.csr_matrix(block),
        obs=observations,
        var=pd.DataFrame(index=genes),
    ).write_h5ad(prediction_path)
    return {"root": root, "config": config_path, "prediction": prediction_path, "targets": root / "targets.json"}


@pytest.fixture
def evaluation_bundle_factory(tmp_path):
    return lambda public_id: make_evaluation_bundle(tmp_path / "evaluation", public_id)
