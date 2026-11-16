"""Generation mathematics, deterministic synthesis and residual-input tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import CELL_LINES, SOURCE_CONFIG

from cellularai.config import Config, load_model
from cellularai.evaluation.scoring import SideStats, expr_mse_parts, group_profile
from cellularai.prediction import placement, plan, residual, synthesis, totals
from cellularai.prediction.inputs import GenerationInputs, residual_summaries
from cellularai.task import Task

MODEL = load_model(SOURCE_CONFIG)


def test_apportion_uses_largest_remainders_and_hits_the_total():
    fractions = np.random.default_rng(0).uniform(0, 1, 50)
    result = synthesis.apportion(fractions, 12345)
    assert result.sum() == 12345
    assert (result >= np.floor(fractions / fractions.sum() * 12345)).all()


def test_synthesis_is_deterministic_and_realizes_integer_gene_totals():
    rng = np.random.default_rng(1)
    controls = rng.poisson(np.r_[rng.uniform(5, 50, 11), 0.01], (500, 12)).astype(np.float32)
    engine = synthesis.Engine(controls, MODEL)
    total = int(round(float(np.full(400, float(engine.depth)).sum())))
    target = synthesis.apportion(engine.design_frac(np.zeros(12)), total)
    libraries = placement.library_profile(controls.sum(axis=1).astype(float), total, MODEL.placement.lib_spread)
    job = dict(
        T=target,
        plan_idx=np.array([0, 1]),
        plan_sign=np.array([1.0, -1.0]),
        plan_priority=np.ones(2, np.float32),
        nudge_sign=np.ones(12, np.int8),
        nudge_strength=np.zeros(12, np.float32),
        rng=placement.pert_rng("gene_00"),
        Lt=libraries,
        mt=np.full(12, -1.0),
    )
    first = synthesis.synthesize(engine, job)
    second = synthesis.synthesize(engine, {**job, "rng": placement.pert_rng("gene_00")})
    assert first.shape == (400, 12)
    assert np.array_equal(first, np.rint(first))
    np.testing.assert_array_equal(first.sum(0)[engine.eligible], target[engine.eligible])
    np.testing.assert_array_equal(first, second)


def test_ratio_correction_preserves_sign_and_bounds_the_factor():
    query = np.array([[1.0, -1.0, 0.5, 0.0]])
    correction = np.array([[10.0, 10.0, -0.1, 3.0]])
    result = residual.bounded_ratio(query, correction)
    np.testing.assert_allclose(result[0, :2], [2.0, -0.5])
    assert result[0, 3] == 0.0 and 0.25 < result[0, 2] < 0.5


def test_sign_preservation_rolls_back_crossings_and_own_target_entries():
    query = np.array([[1.0, -1.0], [0.5, 0.2]])
    adjusted = np.array([[-1.0, -2.0], [0.7, 0.9]])
    result = residual.preserve_signs(query, adjusted, ["gene_a", "gene_b"], ["gene_a", "gene_b"])
    np.testing.assert_array_equal(result, [[1.0, -2.0], [0.7, 0.2]])


def test_evidence_agreement_uses_profile_only_where_totals_agree():
    pooled = np.array([[1.0, 1.0, -1.0]])
    total = np.array([[1.0, 0.5, 1.0]])
    profile = np.array([[2.0, 2.0, -2.0]])
    result = residual.evidence_agreement(pooled, total, profile)
    assert result[0, 0] == 2.0 and 1.0 < result[0, 1] < 2.0 and result[0, 2] == -1.0


def test_response_basis_replaces_own_entries_with_off_target_mean():
    matrix = np.array([[10.0, 2.0, 4.0], [1.0, 20.0, 5.0], [2.0, 4.0, 30.0]])
    _, intercept, centered = residual.response_basis(matrix, np.array([0]), np.array([0]), 2)
    assert intercept[0] == 3.0
    np.testing.assert_allclose(centered[0], [0.0, -1.0, 1.0])


def test_control_log_cpm_normalizes_relative_composition():
    counts = np.array([[10.0, 30.0, np.nan], [0.0, 0.0, np.nan]])
    values, valid = residual.counts_log_cpm(counts)
    np.testing.assert_allclose(values[0, :2], np.log1p([250000.0, 750000.0]))
    assert np.isnan(values[0, 2]) and valid.tolist() == [True, False]


def test_capped_weights_normalize_and_handle_too_few_entries():
    weights = plan.cap_weights([10.0, 1.0, 1.0, 1.0], 0.3)
    assert weights.sum() == pytest.approx(1.0) and weights.max() <= 0.3 + 1e-12
    np.testing.assert_allclose(plan.cap_weights([3.0, 1.0], 0.3), 0.5)


def test_size_curve_has_the_requested_median_and_respects_capacity():
    breadth = np.linspace(-1, 1, 101)
    sizes = plan.curve_sizes(breadth, np.full(101, 1000), MODEL, 400.0)
    assert np.median(sizes) == 400 and sizes.min() >= 5 and sizes.max() <= 1000


def test_power_sizes_reassign_by_score_and_shrink_the_weak_tail():
    initial = np.array([100, 200, 300, 400, 500])
    score = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    result = plan.s1_sizes(initial, score, MODEL)
    assert result[0] > result[1] > result[2]
    assert result[4] == 10


def test_no_evidence_produces_no_placement_targets():
    context = SimpleNamespace(
        l2=np.full((5, 1), 0.1),
        z=np.zeros((5, 1)),
        cpm=np.full(5, 50.0),
        lq=np.zeros((5, 1)),
        elig=np.ones(5, bool),
        depth=20000.0,
        log2_cg=np.zeros(5),
        amp=1.0,
    )
    assert np.isnan(placement.dn_target(context, 0, MODEL)).all()


def test_planned_sign_has_a_minimum_magnitude():
    context = SimpleNamespace(cpm=np.full(3, 100.0), depth=20000.0)
    result = placement.plan_sign(np.array([0.5, 0.5, 0.5]), np.array([0, 1]), np.array([1.0, -1.0]), context, MODEL)
    assert result[0] == 0.5 and -0.5 < result[1] < 0 and result[2] == 0.5


def test_flat_design_has_zero_spread():
    fractions = np.full(4, 0.25)
    spread = totals.make_spread(fractions, totals.own_mask((4, 3), [0, 1, -1]), 10.0)
    assert spread(np.zeros((4, 3))) == 0.0 and spread(np.eye(4)[:, :3]) > 0


def test_integer_totals_spread_equals_design_spread_for_proportional_profiles():
    rng = np.random.default_rng(2)
    fractions = rng.uniform(0.5, 2, 40)
    fractions /= fractions.sum()
    mask = totals.own_mask((40, 6), [0, 1, 2, -1, -1, -1])
    effects = rng.normal(0, 0.5, (40, 6))
    profiles = fractions[:, None] * 2.0**effects
    integer_totals = (profiles / profiles.sum(0) * 1e12).T
    expected = totals.make_spread(fractions, mask, 10.0)(effects)
    assert np.isclose(totals.totals_spread(integer_totals, mask, np.ones(6)), expected, rtol=1e-9)


def test_margin_solver_hits_realized_integer_target():
    rng = np.random.default_rng(4)
    fractions = rng.uniform(0.5, 2, 60)
    fractions /= fractions.sum()
    mask = totals.own_mask((60, 8), list(range(8)))
    spread = totals.make_spread(fractions, mask, 10.0)
    base = rng.normal(0, 1, (60, 8))
    credit = spread(base)

    def realized(margin):
        amplitude, _ = totals.bisect_amp(spread, lambda value: value * base, margin * credit)
        profiles = fractions[:, None] * 2.0 ** (amplitude * base)
        values = np.stack([synthesis.apportion(profiles[:, index], 3000) for index in range(8)])
        return totals.totals_spread(values, mask, np.ones(8))

    target = 0.99 * credit
    margin, reached = totals.solve_margin(realized, target, 0.99)
    assert reached and abs(realized(margin) / target - 1) <= totals.SOLVE_RTOL
    assert abs(margin - 0.99) > 1e-3


def test_margin_solver_uses_bisection_when_secant_stalls():
    margin, reached = totals.solve_margin(lambda value: max(2.0, value + 0.5), 3.0, 0.99)
    assert reached and abs(margin - 2.5) <= 3.0 * totals.SOLVE_RTOL


def test_margin_solver_clamps_unreachable_targets():
    assert totals.solve_margin(lambda value: 5.0 + value, 1.0, 0.99) == (0.0, False)
    margin, reached = totals.solve_margin(lambda value: min(value, 2.0), 3.0, 0.99)
    assert not reached and margin >= 0.99 * 2**totals.SOLVE_STEPS


def test_margin_solver_does_not_claim_success_after_a_jump():
    margin, reached = totals.solve_margin(lambda value: 1.0 if value < 1.5 else 5.0, 3.0, 0.99)
    assert not reached and abs(margin - 1.5) < 1e-6


def test_expected_credit_takes_the_minimum_per_draw():
    estimate = np.array([1.0, 5.0])
    weights = np.array([0.5, 0.25])
    assert totals.expected_credit(estimate, np.array([2.0, 4.0]), weights) == np.mean([1.0, 1.5])
    assert totals.expected_credit(np.full(2, np.inf), np.array([2.0, 4.0]), weights) == 0.75 * 3.0


def test_totals_spread_matches_scorer_spread_component():
    rng = np.random.default_rng(5)
    genes = [f"gene_{index}" for index in range(20)]
    perts = ["gene_0", "gene_3", "target_extra"]
    values = rng.integers(1, 50, (3, 20))
    target_indices = [0, 3, -1]
    weights = 1.0 / (20 - (np.array(target_indices) >= 0))
    side = SideStats(
        perts=perts,
        genes=genes,
        profiles={target: group_profile(values[index][None]) for index, target in enumerate(perts)},
        control_profile=np.zeros(20),
        de={},
        jackknife={target: 1.0 for target in perts},
    )
    scorer_value = expr_mse_parts(side, side)["spread_B"]
    assert np.isclose(
        totals.totals_spread(values, totals.own_mask((20, 3), target_indices), weights), scorer_value, rtol=1e-12
    )


def test_designs_remain_finite_beyond_unbounded_exponent_range():
    fractions = np.array([0.5, 0.5])
    effects = np.array([[2000.0], [0.0]])
    assert np.isfinite(synthesis.Engine.design_frac(SimpleNamespace(fc=fractions, m=MODEL), effects[:, 0])).all()
    assert np.isfinite(totals.make_spread(fractions, np.ones((2, 1), bool), 1000.0)(effects))


def test_library_profile_ignores_empty_control_cells():
    libraries = np.array([0.0, 0.0, 100.0, 200.0])
    np.testing.assert_array_equal(
        placement.library_profile(libraries, 4000, 1.0),
        placement.library_profile(libraries[2:], 4000, 1.0),
    )


def test_no_eligible_genes_produces_empty_call_sets():
    calls = plan.select_calls(
        np.ones((5, 2)),
        np.ones((5, 2)),
        np.full(5, MODEL.calls.eligible_cpm),
        [f"gene_{index}" for index in range(5)],
        ["target_a", "target_b"],
        1e4,
        MODEL,
        median_calls=10,
    )
    assert all(value["n_calls"] == 0 for value in calls.values())


def test_amplitude_search_reports_success_and_cap():
    def spread(values):
        return float(min(values.sum(), 10.0))

    assert totals.bisect_amp(spread, lambda value: np.full(2, value), 100.0) == (64.0, True)
    amplitude, capped = totals.bisect_amp(spread, lambda value: np.full(2, value), 3.0)
    assert not capped and abs(amplitude - 1.5) < 1e-6


def test_shaping_search_reports_its_cap():
    rng = np.random.default_rng(0)
    fractions = rng.uniform(0.5, 2, 30)
    fractions /= fractions.sum()
    mask = totals.own_mask((30, 4), [0, 1, -1, -1])
    spread = totals.make_spread(fractions, mask, 10.0)
    effects = rng.normal(0, 0.3, (30, 4))
    basis = np.linalg.qr(rng.normal(size=(30, 2)))[0].T
    arguments = (effects, fractions, mask, basis, 0.5, np.ones(4), spread)
    assert not totals.cp_design(*arguments, spread(effects))[1]
    assert totals.cp_design(*arguments, 1e12)[1]


def test_synthesis_refuses_inconsistent_library_totals():
    engine = synthesis.Engine(np.random.default_rng(0).poisson(20.0, (50, 8)).astype(float), MODEL)
    job = dict(
        T=np.full(8, 10),
        plan_idx=np.array([], int),
        plan_sign=np.array([]),
        plan_priority=np.array([]),
        nudge_sign=np.zeros(8),
        nudge_strength=np.zeros(8),
        rng=np.random.default_rng(1),
        Lt=np.full(400, 1.0),
        mt=np.full(8, -1.0),
    )
    with pytest.raises(ValueError, match="target libraries sum"):
        synthesis.synthesize(engine, job)


def test_forkserver_synthesis_matches_serial_output(monkeypatch):
    monkeypatch.setenv("TMPDIR", "/dev/shm")
    monkeypatch.setattr(tempfile, "tempdir", None)
    engine = synthesis.Engine(np.random.default_rng(0).poisson(20.0, (50, 8)).astype(float), MODEL)

    def jobs():
        return [
            dict(
                T=np.arange(8) + 5 * index,
                plan_idx=np.array([1], int),
                plan_sign=np.array([1.0]),
                plan_priority=np.array([1.0]),
                nudge_sign=np.ones(8),
                nudge_strength=np.full(8, 0.5),
                rng=np.random.default_rng(index),
                Lt=np.full(400, (np.arange(8) + 5 * index).sum() / 400),
                mt=np.full(8, -1.0),
            )
            for index in range(4)
        ]

    contexts = []
    real_context = synthesis.mp.get_context
    monkeypatch.setattr(synthesis.mp, "get_context", lambda method: contexts.append(method) or real_context(method))
    serial = synthesis.synthesize_all(engine, jobs(), 1)
    parallel = synthesis.synthesize_all(engine, jobs(), 2)
    assert contexts == ["forkserver"]
    assert all(
        first.dtype == second.dtype and first.tobytes() == second.tobytes() for first, second in zip(serial, parallel)
    )


def test_worker_shared_memory_uses_configured_temp(monkeypatch, tmp_path):
    engine = synthesis.Engine(np.random.default_rng(0).poisson(20.0, (50, 8)).astype(float), MODEL)
    job = dict(
        T=np.arange(8) + 1,
        plan_idx=np.array([1], int),
        plan_sign=np.array([1.0]),
        plan_priority=np.array([1.0]),
        nudge_sign=np.ones(8),
        nudge_strength=np.full(8, 0.5),
        rng=np.random.default_rng(0),
        Lt=np.full(400, (np.arange(8) + 1).sum() / 400),
        mt=np.full(8, -1.0),
    )
    seen = []
    original = synthesis.tempfile.TemporaryDirectory
    monkeypatch.setattr(synthesis.tempfile, "TemporaryDirectory", lambda dir: seen.append(dir) or original(dir=dir))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    synthesis.synthesize_all(engine, [job, job], 2)
    assert seen == [str(tmp_path)]


def test_generation_inputs_share_cell_line_axes(world):
    config = Config.load(world["config"])
    task = Task.load(world["task"], config.training_data)
    inputs = GenerationInputs.load(config.training_data, task.genes, task.perts, 4, threads=1)
    assert inputs.breadth_lines == inputs.mix_lines == inputs.sim_lines == CELL_LINES
    assert sorted(inputs.reference_ctrl) == CELL_LINES
    assert inputs.mix_z.shape[0] == inputs.sim_z.shape[0] == inputs.residual_stack.shape[0] == 3
    assert np.isfinite(inputs.residual_counts).all()


def test_mixture_uses_explicit_file_order_not_cell_line_order(world):
    config = Config.load(world["config"])
    task = Task.load(world["task"], config.training_data)
    catalog_path = Path(config.training_data["catalog"])
    payload = json.loads(catalog_path.read_text())
    payload["orders"]["effect_files"] = list(reversed(payload["orders"]["effect_files"]))
    catalog_path.write_text(json.dumps(payload))
    inputs = GenerationInputs.load(config.training_data, task.genes, task.perts, 4, threads=1)
    assert inputs.mix_lines == list(reversed(CELL_LINES))
    assert inputs.breadth_lines == inputs.sim_lines == CELL_LINES


def test_residual_target_selection_is_content_ordered_and_column_order_independent(world):
    config = Config.load(world["config"])
    task = Task.load(world["task"], config.training_data)
    first = residual_summaries(config.training_data, task.genes, 4, ())
    pool = {"gene_01", "gene_03", "gene_05", "gene_07", "gene_09", "target_extra"}
    expected = sorted(sorted(pool, key=lambda name: (hashlib.sha256(name.encode()).digest(), name))[:4])
    assert first[2] == expected

    path = config.training_data and __import__("cellularai.pooling.catalog", fromlist=["Catalog"]).Catalog.load(
        config.training_data
    ).primary_row("cell_line_A").path("effect")
    import pandas as pd

    frame = pd.read_parquet(path)
    frame[["feature_id", *list(frame.columns[:0:-1])]].to_parquet(path, index=False)
    second = residual_summaries(config.training_data, task.genes, 4, ())
    assert second[2] == first[2]
    np.testing.assert_array_equal(second[0][0], first[0][0])
    excluded = residual_summaries(config.training_data, task.genes, 4, ("cell_line_B",))
    assert excluded[0].shape[0] == 2 and excluded[2] == expected
