"""Pooling arithmetic, weighting, ordering and input exclusion tests."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import CELL_LINES, GENES, PERTURBATIONS

from cellularai.config import Config
from cellularai.pooling.__main__ import main as pool_main
from cellularai.pooling.build import Evidence
from cellularai.pooling.catalog import (
    OffsetTable,
    apply_offsets,
    inverse_variance_weights,
    load_null_variance,
    readout_coverage,
    signed_z,
)
from cellularai.pooling.meta_analysis import (
    combine_layer,
    dl_tau2,
    inv_mad_dl,
    parse_op,
    to_l2,
    with_fallback,
)
from cellularai.pooling.weights import CellLineFactors, Features, sim_weight
from cellularai.prediction.inputs import GenerationInputs
from cellularai.task import Task, control_cpm_mean


def table(effect_center=(0.0, 0.0), score_center=(0.0, 0.0), variance=(1.0, 1.0), genes=("gene_a", "gene_b")):
    size = len(genes)
    return OffsetTable(
        cell_line="cell_line_A",
        view="view_0",
        genes=np.asarray(genes),
        effect_center=np.asarray(effect_center, float),
        effect_count=np.full(size, 10.0),
        signed_score_center=np.asarray(score_center, float),
        response_variance=np.asarray(variance, float),
    )


def operations(**changes):
    values = dict(l2_row="1", l2_col="1", l2_mat="auxiliary", z_row="1", z_col="1", z_mat="auxiliary")
    return values | changes


def test_offsets_preserve_missing_values_and_ignore_unavailable_centers():
    panel = np.array([[1.0, 1.0, np.nan], [2.0, 2.0, 2.0]])
    result = apply_offsets(panel, ["gene_a", "gene_b"], table(effect_center=(0.25, np.nan)), "l2")
    np.testing.assert_array_equal(result[0, :2], [0.75, 0.75])
    assert np.isnan(result[0, 2])
    np.testing.assert_array_equal(result[1], panel[1])


def test_signed_scores_keep_sign_and_precision_for_tiny_probabilities():
    result = signed_z(np.array([-1.0, 2.0, np.nan]), np.array([1e-20, 1e-20, 0.5]))
    assert result[0] < -9 and result[1] > 9 and np.isfinite(result[:2]).all()
    assert np.isnan(result[2])


def test_variance_fallback_uses_gene_then_global_medians():
    values = np.array([[1.0, np.nan, np.nan], [3.0, 2.0, np.nan], [5.0, 0.0, np.nan]])
    filled, medians = with_fallback(values)
    np.testing.assert_array_equal(medians, [3.0, 2.0, 2.5])
    np.testing.assert_array_equal(filled[:, 1], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(filled[:, 2], [2.5, 2.5, 2.5])


def test_between_row_variance_matches_the_closed_form():
    effects = [np.array([[1.0, 0.0]]), np.array([[3.0, np.nan]])]
    weights = np.array([[1.0], [2.0]])
    result = dl_tau2(weights, lambda index: (effects[index], effects[index]), 2, 1, 2)
    mean = 7.0 / 3.0
    q_value = (1.0 - mean) ** 2 + 2.0 * (3.0 - mean) ** 2
    c_value = 3.0 - 5.0 / 3.0
    np.testing.assert_allclose(result, [(q_value - 1.0) / c_value])
    assert dl_tau2(weights, lambda index: (effects[0], effects[0]), 2, 1, 2)[0] == 0.0


def test_inverse_variance_weights_add_heterogeneity_and_fill_missing_estimates():
    effects = [np.array([[1.0], [0.0]]), np.array([[3.0], [0.0]])]
    tables = [table(variance=(1.0, np.nan)), table(variance=(0.5, 2.0))]
    result = inv_mad_dl(tables, ["gene_a", "gene_b"], lambda index: (effects[index], effects[index]), 2, 1, 0.003)
    np.testing.assert_array_equal(result[:, 1], [1.0, 1.0])
    variances = np.array([1.0, 0.5])
    tau = dl_tau2(1.0 / variances[:, None], lambda index: (effects[index][:1], effects[index][:1]), 2, 1, 1)[0]
    np.testing.assert_allclose(result[:, 0], 0.75 / (variances + tau))


def test_auxiliary_multiplier_reaches_both_effect_and_score_streams():
    effects = [np.array([[1.0]]), np.array([[4.0]])]
    probabilities = [np.array([[0.01]]), np.array([[0.001]])]
    tables = [table(genes=["gene_a"], effect_center=[0.0], score_center=[0.0], variance=[1.0])] * 2
    effect, score = combine_layer(effects, probabilities, tables, ["gene_a"], operations(), [1.0, 2.0], 0.003)
    np.testing.assert_allclose(effect, [[3.0]])
    first, second = signed_z(np.array([1.0, 4.0]), np.array([0.01, 0.001]))
    np.testing.assert_allclose(score, [[(first + 2.0 * second) / np.sqrt(5.0)]])


def test_missing_effect_is_zero_filled_while_missing_score_is_dropped():
    effects = [np.array([[2.0]]), np.array([[np.nan]])]
    probabilities = [np.array([[0.05]]), np.array([[np.nan]])]
    tables = [table(genes=["gene_a"], effect_center=[0.0], score_center=[0.5], variance=[1.0])] * 2
    effect, score = combine_layer(effects, probabilities, tables, ["gene_a"], operations(), [1.0, 1.0], 0.003)
    np.testing.assert_allclose(effect, [[1.0]])
    expected = abs(signed_z(np.array([2.0]), np.array([0.05]))[0] - 0.5)
    np.testing.assert_allclose(score, [[expected]])


def test_score_stream_uses_its_own_weights_and_raw_effect_sign():
    tables = [table(genes=["gene_a"], effect_center=[0.3], score_center=[0.5], variance=[1.0])]
    factors = CellLineFactors.on_rows(
        ["cell_line_A"],
        ["gene_a"],
        ["cell_line_A"],
        ["gene_a"],
        1,
        row={"nr": np.array([[3.0]])},
    )
    effect, score = combine_layer(
        [np.array([[0.1]])],
        [np.array([[0.2]])],
        tables,
        ["gene_a"],
        operations(z_row="nr"),
        [1.0],
        0.003,
        factors,
    )
    np.testing.assert_allclose(effect, [[-0.2]])
    expected = abs(signed_z(np.array([0.1]), np.array([0.2]))[0] - 0.5)
    np.testing.assert_allclose(score, [[expected]])
    assert score[0, 0] == abs(score[0, 0])


def test_operation_parser_rejects_unknown_and_duplicate_factors():
    assert parse_op("auxiliary*sim*pair", "mat") == ["auxiliary", "sim", "pair"]
    assert parse_op("1", "col") == []
    for expression, axis in (("unknown", "row"), ("kd", "row"), ("sef*sef", "col")):
        with pytest.raises(ValueError):
            parse_op(expression, axis)


def test_effect_conversion_handles_nonfinite_values_and_lower_bound():
    np.testing.assert_array_equal(to_l2(np.array([np.nan, -100.0, 0.5])), [0.0, np.log2(1e-12), 0.5])


def test_accumulation_order_is_explicit_and_observable():
    effects = [np.array([[value]]) for value in (1e16, -1e16, 1.0)]
    probabilities = [np.array([[1.0]])] * 3
    tables = [table(genes=["gene_a"], effect_center=[0.0], score_center=[0.0], variance=[1.0])] * 3
    forward = combine_layer(effects, probabilities, tables, ["gene_a"], operations(), [1.0] * 3, 0.003)[0]
    reverse = combine_layer(effects[::-1], probabilities, tables, ["gene_a"], operations(), [1.0] * 3, 0.003)[0]
    np.testing.assert_allclose(forward, [[1.0 / 3.0]])
    np.testing.assert_array_equal(reverse, [[0.0]])


def test_similarity_weight_is_an_exponential_standardized_correlation():
    correlations = np.array([0.1, 0.5, 0.9])
    expected = np.exp(0.25 * (correlations - correlations.mean()) / correlations.std())
    np.testing.assert_allclose(sim_weight(correlations, 0.25, 0.05, 6.5), expected)


def make_features(world, exclude=()):
    config = Config.load(world["config"])
    task = Task.load(world["task"], config.training_data)
    mean = control_cpm_mean(task.controls("context_A"))
    return Features(config.training_data, task.perts, task.genes, mean, exclude)


def test_knockdown_weight_uses_the_target_gene_effect(world):
    features = make_features(world)
    features.knockdown = np.array([[0.9, 0.3, -0.5, np.nan]])
    np.testing.assert_allclose(features.kd_factor(0.6, 0.0), [[1.0, 0.5, 0.0, 1.0]])
    features = make_features(world)
    line = features.lines.index("cell_line_A")
    target = features.perts.index("gene_01")
    gene = features.union.index("gene_01")
    expected = 1.0 - 2.0 ** features.stack[line, gene, target]
    assert features.knockdown[line, target] == pytest.approx(expected)


def test_pair_weight_halves_only_members_of_a_pair(world):
    features = make_features(world)
    np.testing.assert_array_equal(features.pair_factor(0.5), [0.5, 0.5, 1.0])
    surviving = make_features(world, exclude=("cell_line_B",))
    np.testing.assert_array_equal(surviving.pair_factor(0.5), [1.0, 1.0])
    alternate = make_features(world, exclude=("cell_line_A",))
    np.testing.assert_array_equal(alternate.pair_factor(0.5), [0.5, 1.0])


def test_scale_factor_matches_median_precision_ratio(world):
    from cellularai.pooling.catalog import Catalog, load_frame

    features = make_features(world)
    result = features.se_factor(1.0, 4.0)
    catalog = Catalog.load(Config.load(world["config"]).training_data)
    path = catalog.primary_row("cell_line_B").path("scale")
    scale = load_frame(str(path), features.perts).to_numpy(float)
    medians = np.nanmedian(scale, axis=0)
    expected = np.clip((np.median(medians) / medians) ** 2, 0.25, 4.0)
    np.testing.assert_allclose(result[features.lines.index("cell_line_B")], expected)


def test_noise_ratio_matches_null_over_measured_precision(world):
    features = make_features(world)
    reference = Config.load(world["config"]).training_data
    variance = load_null_variance(reference, features.lines, features.union)
    ratio = (1.0 / variance) / inverse_variance_weights(features.stack)
    np.testing.assert_allclose(features.noise_ratio(1.0, 1e6), ratio / np.median(ratio))


def test_secondary_rows_reuse_primary_factors_and_unknown_rows_are_neutral():
    factors = CellLineFactors.on_rows(
        ["cell_line_A", "cell_line_B"],
        ["gene_a", "gene_b"],
        ["cell_line_A", "cell_line_B", "cell_line_A", "cell_line_B", "cell_line_auxiliary"],
        ["gene_a", "gene_b", "gene_extra"],
        2,
        row={"nr": np.array([[1.0, np.nan], [3.0, 1.0]])},
        col={"kd": np.array([[0.5, 1.0], [1.0, 0.25]])},
        mat={"sim": np.array([2.0, 1.0]), "pair": np.array([0.5, 1.0])},
    )
    names = {"nr", "kd", "sim", "pair"}
    np.testing.assert_array_equal(factors.block(2, names), factors.block(0, names))
    np.testing.assert_array_equal(factors.block(0, names), [[0.5, 1.0], [1.0, 1.0], [1.0, 1.0]])
    assert factors.block(4, names) is None


def test_excluded_summary_files_are_never_opened(world, tmp_path):
    config_path = world["config"]
    task_path = world["task"]
    import yaml

    task_payload = yaml.safe_load(task_path.read_text())
    task_payload["exclude_cell_lines"] = ["cell_line_B"]
    task_path.write_text(yaml.safe_dump(task_payload, sort_keys=False))
    for path in (world["catalog"].parent / "summaries").glob("*cell_line_B*"):
        path.write_bytes(b"not a parquet file")
    for path in (world["catalog"].parent / "offsets").glob("*cell_line_B*"):
        path.write_bytes(b"not an npz file")
    output = tmp_path / "pooled.npz"
    pool_main(["--config", str(config_path), "--task", str(task_path), "--context", "context_A", "--out", str(output)])
    config = Config.load(config_path)
    task = Task.load(task_path, config.training_data)
    evidence = Evidence.load(output, task.genes, task.perts)
    assert np.isfinite(evidence.l2).all() and np.isfinite(evidence.z).all()
    inputs = GenerationInputs.load(config.training_data, task.genes, task.perts, 4, task.exclude, threads=1)
    assert "cell_line_B" not in inputs.breadth_lines
    assert "cell_line_B" not in inputs.mix_lines
    assert "cell_line_B" not in inputs.sim_lines
    assert inputs.residual_stack.shape[0] == len(CELL_LINES) - 1


def test_centering_identifiers_must_match_catalog_row(world):
    from cellularai.pooling.catalog import Catalog, load_tables

    reference = Config.load(world["config"]).training_data
    catalog = Catalog.load(reference)
    row = catalog.rows[0]
    with np.load(row.offset_path(), allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["cell_line"] = np.asarray("mismatched_record")
    np.savez(row.offset_path(), **values)
    with pytest.raises(ValueError, match="centering table identifiers"):
        load_tables(reference, catalog, [row])


def test_absolute_summary_payload_paths_are_refused(world):
    from cellularai.pooling.catalog import Catalog

    catalog = Catalog.load(Config.load(world["config"]).training_data)
    row = catalog.rows[0]
    object.__setattr__(row, "files", {**row.files, "effect": "/absolute/effect.parquet"})
    with pytest.raises(ValueError, match="inside the bundle"):
        row.path("effect")


def test_readout_coverage_lists_missing_and_normalized_near_matches(world):
    from cellularai.pooling.build import training_rows
    from cellularai.pooling.catalog import Catalog

    config = Config.load(world["config"])
    catalog = Catalog.load(config.training_data)
    rows = training_rows(catalog, config.model.pooling)
    genes = [*GENES[:-1], "gene_extra", "GENE_00.2"]
    coverage = readout_coverage(genes, rows)
    assert coverage["missing"] == ["gene_extra", "GENE_00.2"]
    assert coverage["near_miss"] == ["GENE_00.2 -> gene_00"]


def test_evidence_round_trip_and_axis_refusal(world, tmp_path):
    config = Config.load(world["config"])
    task = Task.load(world["task"], config.training_data)
    output = tmp_path / "evidence.npz"
    Evidence(
        np.ones((len(task.genes), len(task.perts))),
        np.ones((len(task.genes), len(task.perts))),
        np.ones((len(task.genes), len(task.perts))),
        np.ones((len(task.genes), len(task.perts))),
        {},
        [],
    ).save(output, task.genes, task.perts)
    loaded = Evidence.load(output, task.genes, task.perts)
    assert loaded.l2.shape == (len(GENES), len(PERTURBATIONS))
    with pytest.raises(ValueError, match="axes differ"):
        Evidence.load(output, task.genes[::-1], task.perts)
