"""Differential expression, metric, bundle-integrity and scoring-flow tests."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pytest
import yaml
from scipy.stats import false_discovery_control, norm, rankdata

from cellularai.evaluation import scoring as scoring
from cellularai.evaluation.data import DATASETS, Inputs, dataset, load_anchors, load_inputs, reference_files
from cellularai.evaluation.de import ControlOrderStats, de_table
from cellularai.evaluation.run import block_from_h5ad
from cellularai.evaluation.run import main as evaluation_main
from cellularai.evaluation.stage import stage
from cellularai.task import Task


def independent_de(counts, control_counts):
    control_cpm = 1e6 * control_counts / control_counts.sum(axis=1, keepdims=True)
    pert_cpm = 1e6 * counts / counts.sum(axis=1, keepdims=True)
    probabilities = []
    fold_changes = []
    n, m = len(counts), len(control_counts)
    for gene in range(counts.shape[1]):
        control = control_cpm[:, gene].astype(np.float32)
        pert = pert_cpm[:, gene].astype(np.float32)
        combined = np.r_[control, pert]
        ranks = rankdata(combined, method="average")
        u_value = ranks[m:].sum() - n * (n + 1) / 2
        _, multiplicities = np.unique(combined, return_counts=True)
        ties = np.sum(multiplicities**3 - multiplicities)
        total = n + m
        variance = (1.0 - ties / (total**3 - total)) * n * m * (total + 1) / 12
        z_value = (u_value - n * m / 2) / np.sqrt(variance) if variance > 0 else 0.0
        probabilities.append(1.0 if variance <= 0 else 2.0 * norm.sf(abs(z_value)))
        fold_changes.append(np.log2((pert_cpm[:, gene].mean() + 1e-9) / (control_cpm[:, gene].mean() + 1e-9)))
    return np.asarray(probabilities), false_discovery_control(probabilities, method="bh"), np.asarray(fold_changes)


def test_de_matches_an_independent_rank_calculation():
    controls = np.array([[10, 20, 5], [20, 10, 7], [15, 15, 4], [18, 13, 8]], dtype=float)
    perturbed = np.array([[30, 10, 5], [25, 15, 4], [28, 12, 7]], dtype=float)
    result = de_table(perturbed, ControlOrderStats.from_control_counts(controls))
    expected_p, expected_adjusted, expected_lfc = independent_de(perturbed, controls)
    np.testing.assert_allclose(result["pval"], expected_p)
    np.testing.assert_allclose(result["padj"], expected_adjusted)
    np.testing.assert_allclose(result["lfc"], expected_lfc)


def test_de_calls_a_shifted_gene_and_gates_low_expression():
    rng = np.random.default_rng(1)
    genes = [f"gene_{index}" for index in range(20)]
    controls = rng.poisson(20.0, (200, len(genes))).astype(np.float32)
    controls[:, 0] = 0
    perturbed = rng.poisson(np.r_[20.0, np.full(len(genes) - 1, 20.0)], (100, len(genes)))
    perturbed[:, 3] = rng.poisson(80.0, 100)
    result = de_table(perturbed, ControlOrderStats.from_control_counts(controls))
    assert result["significant"][3] and result["lfc"][3] > 1.5
    assert not result["eligible"][0] and np.isnan(result["pval"][0]) and not result["significant"][0]


def test_all_tied_gene_has_unit_probability():
    controls = np.full((4, 2), 10.0)
    result = de_table(np.full((3, 2), 10.0), ControlOrderStats.from_control_counts(controls))
    np.testing.assert_array_equal(result["pval"], [1.0, 1.0])
    assert not result["significant"].any()


def test_jackknife_moment_matches_explicit_delete_one_profiles():
    counts = np.random.default_rng(9).poisson(10.0, (6, 4))
    profiles = []
    pooled = counts.sum(axis=0).astype(float)
    total = float(pooled.sum())
    for index in range(len(counts)):
        profiles.append(np.log1p(scoring.TS * (pooled - counts[index]) / (total - counts[index].sum())))
    profiles = np.asarray(profiles)
    expected = (len(counts) - 1) / len(counts) * ((profiles - profiles.mean(axis=0)) ** 2).sum()
    assert scoring.jackknife_moment(counts) == pytest.approx(expected)


def scoring_world():
    rng = np.random.default_rng(0)
    genes = [f"gene_{index:02d}" for index in range(48)]
    perts = ["gene_00", "gene_01", "gene_02", "target_extra"]
    controls = rng.poisson(20.0, (300, len(genes))).astype(np.float32)
    truth = {}
    for index, target in enumerate(perts):
        rates = np.full(len(genes), 20.0)
        start = 4 + 10 * index
        rates[start : start + 12] = 90.0
        truth[target] = rng.poisson(rates, (100, len(genes))).astype(np.float32)
    reference = scoring.build_prediction_side(truth, controls, genes, perts)
    reference.control_jackknife = scoring.jackknife_moment(controls)
    return rng, controls, genes, perts, truth, reference


def test_perfect_prediction_reaches_metric_fixed_points():
    _, controls, genes, perts, truth, reference = scoring_world()
    prediction = scoring.build_prediction_side(truth, controls, genes, perts)
    raw = scoring.score_panel(prediction, reference, [0, 1, 2])
    assert raw["pds"] == 1.0
    assert raw["fid"] == raw["jac"] == raw["reach"] == 1.0
    assert raw["nmae"] == 0.0
    assert raw["expr_mse"] < 0


def test_control_paste_is_uninformative():
    rng, controls, genes, perts, _, reference = scoring_world()
    pasted = {target: controls[rng.choice(300, 100, replace=False)] for target in perts}
    prediction = scoring.build_prediction_side(pasted, controls, genes, perts)
    raw = scoring.score_panel(prediction, reference, [0, 1, 2])
    assert raw["pds"] < 0.8 and raw["fid"] < 0.5


def test_expression_error_is_one_consistent_ratio():
    rng, controls, genes, perts, truth, reference = scoring_world()
    noisy = {target: rng.poisson(np.asarray(values) + 1.0).astype(np.float32) for target, values in truth.items()}
    prediction = scoring.build_prediction_side(noisy, controls, genes, perts)
    parts = scoring.expr_mse_parts(prediction, reference)
    assert parts["expr_mse"] == parts["num"] / parts["den"]
    assert 0 < parts["rho"] <= 1


def test_direction_metrics_match_small_hand_calculations():
    genes = [f"gene_{index}" for index in range(12)]
    reference = {
        "significant": np.ones(12, bool),
        "lfc": np.ones(12),
        "pval": np.linspace(0.001, 0.01, 12),
        "padj": np.linspace(0.001, 0.01, 12),
    }
    prediction = {
        "significant": np.ones(12, bool),
        "lfc": np.array([1.0] * 10 + [-1.0] * 2),
        "pval": np.linspace(0.001, 0.01, 12),
        "padj": np.linspace(0.001, 0.01, 12),
    }
    assert scoring.direction_fidelity(prediction, reference, -1) == pytest.approx(10 / 12)
    assert scoring.direction_reach(prediction, reference, -1, genes) == pytest.approx(11 / 12)
    assert scoring.sig_jaccard(prediction, reference, -1) == 1.0
    assert scoring.lfc_nmae(prediction, reference, -1) == pytest.approx(4 / 12)


def test_jaccard_handles_empty_and_overlapping_sets():
    reference = dict(significant=np.array([False, True, True]), lfc=np.ones(3))
    prediction = dict(significant=np.array([True, True, False]), lfc=np.ones(3))
    assert scoring.sig_jaccard(prediction, reference, -1) == pytest.approx(1 / 3)
    empty = dict(significant=np.zeros(3, bool), lfc=np.ones(3))
    assert scoring.sig_jaccard(empty, empty, -1) == 1.0


def test_anchor_scaling_maps_zero_one_and_clamps_only_selected_metrics():
    baseline = dict(pds=0.5, expr_mse=1.0, fid=0.2, reach=0.1, jac=0.1, nmae=1.0)
    replicate = dict(pds=0.9, expr_mse=0.0, fid=0.8, reach=0.9, jac=0.5, nmae=0.4)
    assert scoring.scaled_scores(baseline, baseline, replicate)["overall"] == 0.0
    assert scoring.scaled_scores(replicate, baseline, replicate)["overall"] == 1.0
    raw = dict(baseline, expr_mse=-1.0, nmae=100.0, pds=1.3)
    result = scoring.scaled_scores(raw, baseline, replicate)
    assert result["expr_mse"] == 1.0 and result["nmae"] == -6.0
    assert result["pds"] == pytest.approx(2.0)


def test_parallel_prediction_side_equals_serial():
    _, controls, genes, perts, truth, _ = scoring_world()
    serial = scoring.build_prediction_side(truth, controls, genes, perts, n_jobs=1)
    parallel = scoring.build_prediction_side(truth, controls, genes, perts, n_jobs=2)
    for target in perts:
        np.testing.assert_array_equal(serial.profiles[target], parallel.profiles[target])
        assert serial.jackknife[target] == parallel.jackknife[target]
        np.testing.assert_array_equal(serial.de[target]["padj"], parallel.de[target]["padj"])


def test_only_public_evaluation_identifiers_are_registered():
    assert set(DATASETS) == {"vcc2025"}
    with pytest.raises(KeyError, match="unknown evaluation dataset"):
        dataset("unknown")


def run_score(bundle, public_id, output, workers="1"):
    evaluation_main(
        [
            "score",
            "--dataset",
            public_id,
            "--config",
            str(bundle["config"]),
            "--prediction",
            str(bundle["prediction"]),
            "--name",
            "synthetic_method",
            "--out",
            str(output),
            "--n-jobs",
            workers,
        ]
    )
    return json.loads((output / "scores.json").read_text())


@pytest.mark.parametrize("public_id", ["vcc2025"])
def test_public_bundle_scores_end_to_end_and_matches_parallel(evaluation_bundle_factory, tmp_path, public_id):
    bundle = evaluation_bundle_factory(public_id)
    serial = run_score(bundle, public_id, tmp_path / "serial")
    parallel = run_score(bundle, public_id, tmp_path / "parallel", workers="2")
    assert parallel == serial
    assert serial["dataset"] == public_id and serial["method"] == "synthetic_method"
    assert serial["raw"]["pds"] == 1.0
    assert serial["raw"]["fid"] == 1.0 and serial["raw"]["nmae"] == 0.0
    assert set(serial) >= {"raw", "scaled", "baseline", "replicate", "mse_parts"}


@pytest.mark.parametrize("axis", ["genes", "perts"])
def test_scoring_requires_exact_reference_axes(evaluation_bundle_factory, tmp_path, axis):
    from cellularai.io import md5

    bundle = evaluation_bundle_factory("vcc2025")
    path = bundle["root"] / "reference.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values[axis] = values[axis][::-1]
    np.savez(path, **values)
    config_path = bundle["config"]
    payload = yaml.safe_load(config_path.read_text())
    payload["datasets"]["vcc2025"]["md5"]["reference_side"] = md5(path)
    config_path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="cached reference axes"):
        run_score(bundle, "vcc2025", tmp_path / "score")


@pytest.mark.parametrize("public_id", ["vcc2025"])
def test_changed_bundle_file_is_refused_by_md5(evaluation_bundle_factory, public_id):
    bundle = evaluation_bundle_factory(public_id)
    bundle["targets"].write_text('["gene_02", "gene_01", "gene_00"]')
    with pytest.raises(ValueError, match="MD5"):
        run_score(bundle, public_id, bundle["root"] / "refused")


@pytest.mark.parametrize("indices", [[0, -1], []])
def test_empty_or_out_of_bounds_control_provision_is_refused(tmp_path, indices):
    np.savez(tmp_path / "controls.npz", counts=np.ones((3, 2)), genes=np.array(["gene_a", "gene_b"]))
    np.save(tmp_path / "provision.npy", np.asarray(indices, np.int64))
    (tmp_path / "targets.json").write_text('["gene_a"]')
    files = {
        "controls": tmp_path / "controls.npz",
        "provision": tmp_path / "provision.npy",
        "perturbations": tmp_path / "targets.json",
    }
    with pytest.raises(ValueError, match="empty or out of bounds"):
        load_inputs(dataset("vcc2025"), files)


def test_unsupported_anchor_formula_is_refused(evaluation_bundle_factory):
    bundle = evaluation_bundle_factory("vcc2025")
    anchor = bundle["root"] / "anchors.json"
    payload = json.loads(anchor.read_text())
    payload["code_version"] = "unknown"
    anchor.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported anchor formula"):
        load_anchors(anchor)


def test_prediction_block_validates_axes_dtype_and_cell_counts(evaluation_bundle_factory):
    bundle = evaluation_bundle_factory("vcc2025")
    selected = dataset("vcc2025")
    inputs = load_inputs(selected, reference_files(yaml.safe_load(bundle["config"].read_text()), selected))
    block = block_from_h5ad(bundle["prediction"], inputs)
    assert block.shape == (3, 400, 48) and block.dtype == np.float32
    prediction = ad.read_h5ad(bundle["prediction"])
    broken = prediction[:100].copy()
    broken.write_h5ad(bundle["root"] / "broken.h5ad")
    with pytest.raises(ValueError, match="perturbations differ|equal cell counts"):
        block_from_h5ad(bundle["root"] / "broken.h5ad", inputs)
    targets = prediction.obs["target_gene"].astype(str).to_numpy()
    first = np.flatnonzero(targets == targets[0])
    uneven = prediction[np.concatenate([first[:-1], np.flatnonzero(targets != targets[0])])].copy()
    uneven.write_h5ad(bundle["root"] / "uneven.h5ad")
    with pytest.raises(ValueError, match="equal cell counts"):
        block_from_h5ad(bundle["root"] / "uneven.h5ad", inputs)
    keep = np.concatenate([np.flatnonzero(targets == target)[:50] for target in inputs.perts])
    prediction[keep].copy().write_h5ad(bundle["root"] / "small.h5ad")
    assert block_from_h5ad(bundle["root"] / "small.h5ad", inputs).shape == (3, 50, 48)


@pytest.mark.parametrize("public_id", ["vcc2025"])
def test_staging_creates_an_ordinary_task_for_the_public_dataset(
    evaluation_bundle_factory, world, tmp_path, public_id
):
    selected = dataset(public_id)
    bundle = evaluation_bundle_factory(public_id)
    config = yaml.safe_load(bundle["config"].read_text())
    inputs = load_inputs(selected, reference_files(config, selected))
    template = yaml.safe_load(world["task"].read_text())
    task_path = stage(inputs, template, tmp_path / "staged")
    payload = yaml.safe_load(task_path.read_text())
    assert payload["round"] == public_id and payload["contexts"] == {public_id: {"controls": "controls.h5ad"}}
    task = Task.load(task_path, {"catalog": str(world["catalog"])})
    assert task.genes == inputs.genes and task.perts == inputs.perts
    assert task.controls(public_id).shape == inputs.controls.shape


def test_relative_staging_output_creates_a_relocatable_task(evaluation_bundle_factory, world, tmp_path, monkeypatch):
    bundle = evaluation_bundle_factory("vcc2025")
    monkeypatch.chdir(tmp_path)
    evaluation_main(
        [
            "stage",
            "--dataset",
            "vcc2025",
            "--config",
            str(bundle["config"]),
            "--like",
            str(world["task"]),
            "--out",
            "relative_staged",
        ]
    )
    task = Task.load(tmp_path / "relative_staged" / "task.yaml", {"catalog": str(world["catalog"])})
    assert task.root == tmp_path / "relative_staged"
    assert task.controls("vcc2025").shape[0] > 0


def test_staging_refuses_fractional_counts(evaluation_bundle_factory, world, tmp_path):
    selected = dataset("vcc2025")
    bundle = evaluation_bundle_factory("vcc2025")
    inputs = load_inputs(selected, reference_files(yaml.safe_load(bundle["config"].read_text()), selected))
    counts = inputs.controls.copy()
    counts[0, 0] = 0.5
    fractional = Inputs(inputs.dataset, inputs.genes, inputs.perts, counts)
    with pytest.raises(ValueError, match="non-negative integers"):
        stage(fractional, yaml.safe_load(world["task"].read_text()), tmp_path / "fractional")
