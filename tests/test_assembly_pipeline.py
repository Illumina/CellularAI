"""Assembly quality gates, h5ad merging and command-construction tests."""

from __future__ import annotations

import dataclasses
import json

import anndata as ad
import numpy as np
import pytest
import yaml
from conftest import SOURCE_CONFIG, clean_prediction

from cellularai import assemble, pipeline
from cellularai.config import Config, load_model
from cellularai.guards import NAMES
from cellularai.task import Task

LIMITS = load_model(SOURCE_CONFIG).guards
PROVENANCE = {"config": "config_identity", "task": "task_identity"}


def clean_report(**changes):
    guards_payload = {name: [0, 1_000_000, np.nan] for name in NAMES}
    guards_payload["nr_clip"] = [0, 4_000_000, 12.0]
    guards_payload.update(changes.pop("guards", {}))
    return {
        "guards": guards_payload,
        "unmeasured": {"no_row": [], "no_evidence": []},
        "readout": {"n_genes": 1000, "missing": [], "near_miss": []},
        "provenance": dict(PROVENANCE),
        "margin_reached": True,
        "converged": True,
        "credit_share": 0.99,
        **changes,
    }


def false_checks(result):
    return [key for key, value in result.items() if isinstance(value, (bool, np.bool_)) and not value]


def test_clean_build_report_passes_every_gate():
    result = assemble.build_gates(clean_report(), LIMITS, PROVENANCE)
    assert false_checks(result) == [] and result["failures"] == []


@pytest.mark.parametrize(
    "changes, failed",
    [
        ({"unmeasured": {"no_row": ["target_a"], "no_evidence": ["target_a", "target_b"]}}, "unmeasured"),
        ({"margin_reached": False}, "margin_reached"),
        ({"converged": False}, "converged"),
        ({"credit_share": 1.001}, "credit_share"),
        ({"credit_share": 0.974}, "credit_band"),
        ({"guards": {"nr_clip": [81, 4_000_000, 300.0]}}, "guard_nr_clip"),
        ({"guards": {"log2_overflow": [1, 1_000_000, np.nan]}}, "guard_log2_overflow"),
        ({"guards": {"l2_abs_max": [1, 1_000_000, 7.0]}}, "guard_l2_abs_max"),
        ({"guards": {"amp_search_cap": [1, 3, np.nan]}}, "guard_amp_search_cap"),
        ({"guards": {"shape_search_cap": [1, 2, np.nan]}}, "guard_shape_search_cap"),
        (
            {"readout": {"n_genes": 1000, "missing": [f"gene_{index}" for index in range(6)], "near_miss": []}},
            "readout_missing",
        ),
        ({"readout": {}}, "readout_missing"),
        ({"provenance": {"config": "other", "task": "task_identity"}}, "provenance"),
        ({"provenance": {}}, "provenance"),
    ],
)
def test_each_build_gate_fails_independently(changes, failed):
    result = assemble.build_gates(clean_report(**changes), LIMITS, PROVENANCE)
    assert false_checks(result) == [failed]


def test_allowed_guard_rate_passes_and_unmeasured_names_are_deduplicated():
    result = assemble.build_gates(
        clean_report(
            guards={"nr_clip": [80, 4_000_000, 300.0]},
            unmeasured={"no_row": ["target_a"], "no_evidence": ["target_a", "target_b"]},
        ),
        dataclasses.replace(LIMITS, max_unmeasured=2),
        PROVENANCE,
    )
    assert result["guard_nr_clip"]
    assert result["unmeasured_perturbations"] == ["target_a", "target_b"]


def test_missing_guard_registration_and_stale_rate_keys_fail():
    report = clean_report()
    del report["guards"]["mt_ls_clip"]
    result = assemble.build_gates(report, LIMITS, PROVENANCE)
    assert false_checks(result) == ["guards_registered"]
    stale = dataclasses.replace(LIMITS, max_hit_rate={**LIMITS.max_hit_rate, "unknown_guard": 1.0})
    assert false_checks(assemble.build_gates(clean_report(), stale, PROVENANCE)) == ["guards_registered"]


def test_readout_gate_accepts_its_limit_and_lists_missing_features():
    readout = {"n_genes": 1000, "missing": [f"gene_{index}" for index in range(5)], "near_miss": ["gene_4 -> gene_04"]}
    result = assemble.build_gates(clean_report(readout=readout), LIMITS, PROVENANCE)
    assert result["readout_missing"] and result["readout_missing_frac"] == 0.005
    assert result["readout_near_miss"] == ["gene_4 -> gene_04"]
    readout["missing"].append("gene_5")
    result = assemble.build_gates(clean_report(readout=readout), LIMITS, PROVENANCE)
    assert not result["readout_missing"] and len(result["readout_missing_genes"]) == 6


def test_prediction_gate_checks_layout_counts_context_and_call_depth(world):
    config = Config.load(world["config"])
    task = Task.load(world["task"], config.training_data)
    expected = {"config": config.fingerprint, "task": task.fingerprint}
    prediction = clean_prediction(world, world["root"] / "unused.h5ad")
    controls = task.controls("context_A")
    result = assemble.gate(
        prediction, "context_A", task.genes, task.perts, controls, task.cfg["call_depth"], LIMITS, expected
    )
    assert false_checks(result) == []
    assert (
        assemble.gate(
            prediction, "context_A", task.genes, task.perts, controls, result["median_n_pred"] - 1, LIMITS, expected
        )["call_depth"]
        is False
    )
    assert (
        assemble.gate(prediction, "context_B", task.genes, task.perts, controls, 30, LIMITS, expected)["context"]
        is False
    )
    assert (
        assemble.gate(prediction[1:], "context_A", task.genes, task.perts, controls, 30, LIMITS, expected)[
            "cells_per_perturbation"
        ]
        is False
    )
    assert (
        assemble.gate(prediction, "context_A", task.genes, task.perts[:-1], controls, 30, LIMITS, expected)[
            "perturbations"
        ]
        is False
    )
    fractional = prediction.copy()
    fractional.X.data[0] += 0.5
    assert (
        assemble.gate(fractional, "context_A", task.genes, task.perts, controls, 30, LIMITS, expected)["integer_counts"]
        is False
    )
    excessive = prediction.copy()
    excessive.X.data[0] = 2**24
    assert (
        assemble.gate(excessive, "context_A", task.genes, task.perts, controls, 30, LIMITS, expected)["integer_counts"]
        is False
    )
    wider = prediction.copy()
    wider.X = wider.X.astype(np.float64)
    assert (
        assemble.gate(wider, "context_A", task.genes, task.perts, controls, 30, LIMITS, expected)["float32_counts"]
        is False
    )
    dense = prediction.copy()
    dense.X = dense.X.toarray()
    assert (
        assemble.gate(dense, "context_A", task.genes, task.perts, controls, 30, LIMITS, expected)["sparse_csr"] is False
    )


def one_context(world):
    payload = yaml.safe_load(world["task"].read_text())
    payload["contexts"] = {"context_A": payload["contexts"]["context_A"]}
    world["task"].write_text(yaml.safe_dump(payload, sort_keys=False))


def test_assembly_merges_predictions_to_h5ad(world, tmp_path):
    one_context(world)
    builds = tmp_path / "builds"
    original = clean_prediction(world, assemble.predicted_part(builds, "context_A"))
    assemble.main(["--config", str(world["config"]), "--task", str(world["task"]), "--builds", str(builds)])
    merged = ad.read_h5ad(builds / "prediction.h5ad")
    assert merged.shape == original.shape
    assert json.loads((builds / "gates.json").read_text())["failed"] == []


def test_failed_gate_stops_before_merging(world, tmp_path):
    one_context(world)
    builds = tmp_path / "builds"
    prediction_path = assemble.predicted_part(builds, "context_A")
    prediction = clean_prediction(world, prediction_path)
    prediction.uns["build"]["margin_reached"] = False
    prediction.write_h5ad(prediction_path)
    with pytest.raises(SystemExit, match="failed quality gates"):
        assemble.main(["--config", str(world["config"]), "--task", str(world["task"]), "--builds", str(builds)])
    gates = json.loads((builds / "gates.json").read_text())
    assert "context_A.margin_reached" in gates["failed"]
    assert not (builds / "prediction.h5ad").exists()


def test_existing_final_artifacts_are_never_overwritten(world, tmp_path):
    one_context(world)
    builds = tmp_path / "builds"
    builds.mkdir()
    (builds / "prediction.h5ad").write_text("older prediction")
    with pytest.raises(SystemExit, match="fresh output directory"):
        assemble.main(["--config", str(world["config"]), "--task", str(world["task"]), "--builds", str(builds)])
    assert (builds / "prediction.h5ad").read_text() == "older prediction"


def test_pipeline_constructs_pool_predict_assemble_and_score_commands(world, tmp_path):
    jobs = pipeline.commands(
        world["config"],
        world["task"],
        tmp_path / "output",
        "python",
        workers=1,
        score="vcc2025",
        evaluation_config=world["root"] / "evaluation.yaml",
    )
    assert set(jobs) == {"ctx_context_A", "ctx_context_B", "assemble"}
    for context in ("context_A", "context_B"):
        pool, predict = jobs[f"ctx_{context}"]
        assert pool[2:4] == ["-m", "cellularai.pooling"]
        assert predict[2:4] == ["-m", "cellularai.prediction"]
        assert predict[-2:] == ["--n-jobs", "1"]
        assert context in pool and context in predict
    assert jobs["assemble"][0][2:4] == ["-m", "cellularai.assemble"]
    score = jobs["assemble"][1]
    assert score[4:6] == ["score", "--dataset"]
    assert "vcc2025" in score and score[-2:] == ["--out", str(tmp_path / "output" / "score")]


def test_score_command_requires_evaluation_configuration(world, tmp_path):
    with pytest.raises(ValueError, match="evaluation-config"):
        pipeline.commands(world["config"], world["task"], tmp_path, "python", score="vcc2025")


def test_slurm_uses_configured_temp_and_preserves_score_config(world, tmp_path):
    jobs = pipeline.commands(
        world["config"],
        world["task"],
        tmp_path / "output",
        "python",
        workers=1,
        score="vcc2025",
        evaluation_config=tmp_path / "evaluation.yaml",
    )
    launch = pipeline.write_slurm(
        jobs,
        tmp_path / "output",
        {"tmp": str(tmp_path / "short_tmp"), "slurm": {"build": "--partition=cpu", "assemble": "--partition=cpu"}},
    )
    script = (tmp_path / "output" / "jobs" / "assemble.sh").read_text()
    assert f"TMPDIR={tmp_path / 'short_tmp'}" in script
    assert str(tmp_path / "evaluation.yaml") in script
    assert launch.exists()


def test_slurm_shell_identifiers_are_independent_of_context_names(world, tmp_path):
    payload = yaml.safe_load(world["task"].read_text())
    payload["contexts"] = {"context-A": payload["contexts"]["context_A"]}
    world["task"].write_text(yaml.safe_dump(payload, sort_keys=False))
    jobs = pipeline.commands(world["config"], world["task"], tmp_path / "output", "python", workers=1)
    launch = pipeline.write_slurm(
        jobs,
        tmp_path / "output",
        {"slurm": {"build": "--partition=cpu", "assemble": "--partition=cpu"}},
    )
    text = launch.read_text()
    assert "job_0=$(sbatch" in text
    assert "$job_0" in text
    assert "ctx_context-A=" not in text
    payload["contexts"] = {"bad;name": {"controls": "context_A_controls.h5ad"}}
    with pytest.raises(ValueError, match="invalid context name"):
        from cellularai.task import check_cfg

        check_cfg(payload, "task")


def test_local_pipeline_uses_configured_temp(tmp_path):
    import sys

    temporary = tmp_path / "short_tmp"
    pipeline.run_local({"step": [[sys.executable, "-c", "pass"]]}, tmp_path / "output", {"tmp": str(temporary)})
    assert temporary.is_dir()
