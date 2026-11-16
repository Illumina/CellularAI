"""Configuration, task-schema, control-matrix and integrity validation tests."""

from __future__ import annotations

import copy
import pickle

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from conftest import GENES, SOURCE_CONFIG

from cellularai.config import Config, load_model
from cellularai.io import load_yaml, read_column
from cellularai.preflight import verify_summary_bundle
from cellularai.task import Task, check_cfg, read_controls


def load_payload_model(tmp_path, payload):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({"model": payload}))
    return load_model(path)


def test_public_model_loads_and_pickles():
    model = load_model(SOURCE_CONFIG)
    assert model.pooling.views == ("view_0", "view_1")
    assert pickle.loads(pickle.dumps(model)) == model


@pytest.mark.parametrize("path", [("totals", "credit_draws"), ("placement", "rule", "cl2"), ("pooling", "sim")])
def test_missing_model_keys_are_refused(tmp_path, model_payload, path):
    payload = copy.deepcopy(model_payload)
    section = payload
    for key in path[:-1]:
        section = section[key]
    del section[path[-1]]
    with pytest.raises(ValueError, match=rf"missing \['{path[-1]}'\]"):
        load_payload_model(tmp_path, payload)


@pytest.mark.parametrize("section_path", ["calls", "placement.rule", None])
def test_unknown_model_keys_are_refused(tmp_path, model_payload, section_path):
    payload = copy.deepcopy(model_payload)
    section = payload
    for key in section_path.split(".") if section_path else []:
        section = section[key]
    section["unrecognized_value"] = 1.0
    with pytest.raises(ValueError, match=r"unknown \['unrecognized_value'\]"):
        load_payload_model(tmp_path, payload)


def test_unsupported_pooling_modes_are_refused(tmp_path, model_payload):
    payload = copy.deepcopy(model_payload)
    payload["pooling"]["fill"] = "other"
    with pytest.raises(ValueError, match="only correction"):
        load_payload_model(tmp_path, payload)


def test_duplicate_yaml_keys_are_refused(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("section:\n  value: 1\n  value: 2\n")
    with pytest.raises(ValueError, match=r"duplicate YAML keys \['value'\]"):
        load_yaml(path)


def test_fingerprints_track_model_and_task_content_not_compute_settings(world):
    initial = Config.load(world["config"]).fingerprint
    payload = yaml.safe_load(world["config"].read_text())
    payload["compute"]["workers"] = 2
    world["config"].write_text(yaml.safe_dump(payload, sort_keys=False))
    assert Config.load(world["config"]).fingerprint == initial
    payload["model"]["guards"]["nr_clip"] += 1.0
    world["config"].write_text(yaml.safe_dump(payload, sort_keys=False))
    assert Config.load(world["config"]).fingerprint != initial

    training_data = {"catalog": str(world["catalog"])}
    task_fingerprint = Task.load(world["task"], training_data).fingerprint
    task_payload = yaml.safe_load(world["task"].read_text())
    task_payload["call_depth"] += 1
    world["task"].write_text(yaml.safe_dump(task_payload, sort_keys=False))
    assert Task.load(world["task"], training_data).fingerprint != task_fingerprint


@pytest.mark.parametrize(
    "change, message",
    [
        ({"median_dges": 8}, r"unknown \['median_dges'\]"),
        ({"median_degs": None}, "median_degs must be a positive number"),
        ({"call_depth": "30"}, "call_depth must be a positive number"),
        ({"call_depth": -1}, "call_depth must be a positive number"),
        ({"call_depth": True}, "call_depth must be a positive number"),
        ({"call_depth": float("nan")}, "call_depth must be a positive number"),
        ({"median_degs": 0}, "median_degs must be a positive number"),
        ({"contexts": {}}, "non-empty mapping"),
        ({"contexts": {False: {"controls": "x"}}}, "context name False"),
        ({"contexts": {"bad,name": {"controls": "x"}}}, "context name 'bad,name'"),
        ({"contexts": {"bad name": {"controls": "x"}}}, "context name 'bad name'"),
        ({"contexts": {"context_A": {"controls": "x", "extra": "y"}}}, "expected exactly"),
        ({"exclude_cell_lines": "cell_line_A"}, "exclude_cell_lines must be a list"),
        ({"control_label": ""}, "control_label must be a non-empty string"),
        ({"cells_per_perturbation": 1}, "cells_per_perturbation must be an integer >= 2"),
        ({"cells_per_perturbation": 100.0}, "cells_per_perturbation must be an integer >= 2"),
        ({"cells_per_perturbation": True}, "cells_per_perturbation must be an integer >= 2"),
        ({"seed": -1}, "seed must be a non-negative integer"),
        ({"seed": 1.5}, "seed must be a non-negative integer"),
        ({"seed": False}, "seed must be a non-negative integer"),
    ],
)
def test_invalid_task_schema_is_refused(world, change, message):
    payload = yaml.safe_load(world["task"].read_text()) | change
    with pytest.raises(ValueError, match=message):
        check_cfg(payload, "synthetic_task")


def test_cells_per_perturbation_defaults_to_400_and_enters_the_fingerprint(world):
    training_data = {"catalog": str(world["catalog"])}
    default = Task.load(world["task"], training_data)
    assert default.n_cells == 400
    payload = yaml.safe_load(world["task"].read_text())
    world["task"].write_text(yaml.safe_dump(payload | {"cells_per_perturbation": 400}, sort_keys=False))
    explicit = Task.load(world["task"], training_data)
    assert explicit.n_cells == 400
    world["task"].write_text(yaml.safe_dump(payload | {"cells_per_perturbation": 50}, sort_keys=False))
    custom = Task.load(world["task"], training_data)
    assert custom.n_cells == 50 and custom.fingerprint != default.fingerprint


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("fixed_steps", 0, "fixed_steps must be a positive integer"),
        ("fixed_steps", 2.0, "fixed_steps must be a positive integer"),
        ("fixed_rtol", 0, "fixed_rtol must be a positive number"),
        ("fixed_rtol", "0.001", "fixed_rtol must be a positive number"),
    ],
)
def test_invalid_fixed_point_settings_are_refused(tmp_path, model_payload, key, value, message):
    payload = copy.deepcopy(model_payload)
    payload["totals"][key] = value
    with pytest.raises(ValueError, match=message):
        load_payload_model(tmp_path, payload)


def test_seed_is_optional_and_enters_the_fingerprint(world):
    training_data = {"catalog": str(world["catalog"])}
    default = Task.load(world["task"], training_data)
    assert default.seed is None
    payload = yaml.safe_load(world["task"].read_text())
    world["task"].write_text(yaml.safe_dump(payload | {"seed": 0}, sort_keys=False))
    seeded = Task.load(world["task"], training_data)
    assert seeded.seed == 0 and seeded.fingerprint != default.fingerprint


def test_missing_task_key_is_refused(world):
    payload = yaml.safe_load(world["task"].read_text())
    del payload["call_depth"]
    with pytest.raises(ValueError, match=r"missing \['call_depth'\]"):
        check_cfg(payload, "synthetic_task")


def test_yaml_boolean_context_name_is_rejected(world):
    text = world["task"].read_text().replace("context_B:", "no:")
    world["task"].write_text(text)
    with pytest.raises(ValueError, match="context name False"):
        Task.load(world["task"], {"catalog": str(world["catalog"])})


def test_duplicate_readout_and_target_names_are_refused(world):
    root = world["task"].parent / "task"
    pd.DataFrame({"gene_name": GENES[:-1] + [GENES[0]]}).to_csv(root / "readout.csv", index=False)
    with pytest.raises(ValueError, match=r"duplicate names \['gene_00'\]"):
        Task.load(world["task"], {"catalog": str(world["catalog"])})


@pytest.mark.parametrize(
    "text, message",
    [
        ("target_gene\ngene_01\n \ngene_03\ngene_05\n", "empty name"),
        ("target_gene,n\ngene_01,1\n,1\ngene_03,1\ngene_05,1\n", "empty name"),
        ("target_gene\ngene_01\nNegative_Control\ngene_03\ngene_05\n", "only by case"),
    ],
)
def test_blank_and_control_like_targets_are_refused(world, text, message):
    (world["task"].parent / "task" / "targets.csv").write_text(text)
    with pytest.raises(ValueError, match=message):
        Task.load(world["task"], {"catalog": str(world["catalog"])})


def test_targets_are_sorted_and_exact_control_label_is_removed(world):
    (world["task"].parent / "task" / "targets.csv").write_text(
        "target_gene\nnegative_control\ngene_05\ngene_01\ngene_03\n"
    )
    task = Task.load(world["task"], {"catalog": str(world["catalog"])})
    assert task.perts == ["gene_01", "gene_03", "gene_05"]


def test_duplicate_targets_after_whitespace_stripping_are_refused(world):
    (world["task"].parent / "task" / "targets.csv").write_text("target_gene\ngene_01\n gene_01\ngene_03\ngene_05\n")
    with pytest.raises(ValueError, match="duplicate perturbations"):
        Task.load(world["task"], {"catalog": str(world["catalog"])})


def test_fewer_than_three_targets_are_refused(world):
    (world["task"].parent / "task" / "targets.csv").write_text("target_gene\ngene_01\ngene_03\n")
    with pytest.raises(ValueError, match="at least three"):
        Task.load(world["task"], {"catalog": str(world["catalog"])})


def test_unknown_excluded_cell_line_is_refused(world):
    payload = yaml.safe_load(world["task"].read_text())
    payload["exclude_cell_lines"] = ["cell_line_unknown"]
    world["task"].write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ValueError, match="cell_line_unknown"):
        Task.load(world["task"], {"catalog": str(world["catalog"])})


def test_unknown_context_is_refused(world):
    task = Task.load(world["task"], {"catalog": str(world["catalog"])})
    with pytest.raises(ValueError, match="not a task context"):
        task.controls("context_unknown")


def replace_controls(world, counts, genes=GENES):
    ad.AnnData(X=sp.csr_matrix(np.asarray(counts, np.float32)), var=pd.DataFrame(index=genes)).write_h5ad(
        world["task"].parent / "task" / "context_A_controls.h5ad"
    )


@pytest.mark.parametrize("rows", [0, 1])
def test_too_few_control_cells_are_refused(world, rows):
    replace_controls(world, np.random.default_rng(0).poisson(10.0, (rows, len(GENES))))
    task = Task.load(world["task"], {"catalog": str(world["catalog"])})
    with pytest.raises(ValueError, match="at least two control cells"):
        task.controls("context_A")


@pytest.mark.parametrize("selection", [slice(None), slice(None, None, 2), slice(0, 1)])
def test_empty_control_cells_are_refused(world, selection):
    counts = np.random.default_rng(0).poisson(10.0, (450, len(GENES)))
    counts[selection] = 0
    replace_controls(world, counts)
    task = Task.load(world["task"], {"catalog": str(world["catalog"])})
    with pytest.raises(ValueError, match="control cells have no counts"):
        task.controls("context_A")


def test_control_feature_order_must_match_task(world):
    replace_controls(world, np.random.default_rng(0).poisson(10.0, (20, len(GENES))), GENES[::-1])
    task = Task.load(world["task"], {"catalog": str(world["catalog"])})
    with pytest.raises(ValueError, match="control genes differ"):
        task.controls("context_A")


def raw_h5(path, data, indices, indptr, shape, encoding="csr_matrix", names=GENES):
    with h5py.File(path, "w") as handle:
        var = handle.create_group("var")
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=np.asarray(names, dtype="S"))
        matrix = handle.create_group("X")
        matrix.attrs["encoding-type"], matrix.attrs["shape"] = encoding, shape
        matrix["data"], matrix["indices"], matrix["indptr"] = np.asarray(data, np.float32), indices, indptr


def test_malformed_sparse_controls_are_refused(tmp_path):
    path = tmp_path / "controls.h5ad"
    raw_h5(path, [1.0], [len(GENES) + 3], [0, 1], (1, len(GENES)))
    with pytest.raises(ValueError):
        read_controls(path)
    raw_h5(path, [1.0], [0], [0, 1], (1, len(GENES)), encoding="coo_matrix")
    with pytest.raises(ValueError, match="X must be CSR or CSC"):
        read_controls(path)
    raw_h5(path, [1.0], [0], [0, 1], (1, len(GENES) + 1))
    with pytest.raises(ValueError, match="matrix and gene-axis dimensions differ"):
        read_controls(path)


@pytest.mark.parametrize("bad", [0.5, -1.0, np.nan, np.inf, 2.0**24])
def test_invalid_control_values_are_refused(tmp_path, bad):
    path = tmp_path / "controls.h5ad"
    raw_h5(path, [bad], [0], [0, 1], (1, len(GENES)))
    with pytest.raises(ValueError, match="non-negative integers"):
        read_controls(path)


def test_duplicate_sparse_entries_are_summed_before_validation(tmp_path):
    path = tmp_path / "controls.h5ad"
    raw_h5(path, [2.0**23, 2.0**23], [0, 0], [0, 2], (1, len(GENES)))
    with pytest.raises(ValueError, match="below 2"):
        read_controls(path)


def test_categorical_features_with_missing_codes_are_refused(tmp_path):
    path = tmp_path / "controls.h5ad"
    raw_h5(path, [1.0], [0], [0, 1], (1, 2), names=["gene_a", "gene_b"])
    with h5py.File(path, "r+") as handle:
        del handle["var"]["_index"]
        index = handle["var"].create_group("_index")
        index["categories"] = np.array(["gene_a", "gene_b"], dtype="S")
        index["codes"] = np.array([0, -1], np.int8)
    with pytest.raises(ValueError, match="missing gene names"):
        read_controls(path)


def test_string_columns_preserve_missing_like_names(tmp_path):
    path = tmp_path / "names.csv"
    path.write_text("gene_name\nNA\nNA.1\n")
    assert read_column(path, "gene_name") == ["NA", "NA.1"]


def test_summary_checksum_verification_passes_and_detects_changes(world):
    verify_summary_bundle(world["catalog"])
    row = __import__("cellularai.pooling.catalog", fromlist=["Catalog"]).Catalog.load(world["catalog"]).rows[0]
    row.path("effect").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity verification"):
        verify_summary_bundle(world["catalog"])


def test_every_referenced_summary_file_must_have_a_checksum(world):
    checksum_path = world["catalog"].parent / "SHA256SUMS"
    lines = checksum_path.read_text().splitlines()
    checksum_path.write_text("\n".join(line for line in lines if "scale.parquet" not in line) + "\n")
    with pytest.raises(ValueError, match="missing from checksum manifest"):
        verify_summary_bundle(world["catalog"])


def test_checksum_manifest_cannot_escape_the_bundle(world):
    checksum_path = world["catalog"].parent / "SHA256SUMS"
    checksum_path.write_text(checksum_path.read_text() + "0" * 64 + "  ../outside.npz\n")
    with pytest.raises(ValueError, match="outside the bundle"):
        verify_summary_bundle(world["catalog"])
