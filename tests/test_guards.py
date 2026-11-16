"""Numerical-bound and concentration-counter tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import SOURCE_CONFIG
from test_pooling import make_features, table

from cellularai import guards
from cellularai.config import load_model
from cellularai.pooling.meta_analysis import inv_mad_dl
from cellularai.pooling.weights import sim_weight
from cellularai.prediction import exp2, placement, plan

LIMITS = load_model(SOURCE_CONFIG).guards


@pytest.fixture(autouse=True)
def fresh_counters():
    guards.reset()
    yield
    guards.reset()


def test_counters_track_hits_values_seen_and_extremes():
    guards.over(np.array([-7.0, -5.0, 0.0, 5.0, np.nan]), 5.0, "over_test")
    guards.over(np.array([6.0]), 5.0, "over_test")
    guards.under(np.array([0.01, 0.05, 0.2]), 0.05, "under_test")
    guards.under(0.001, 0.05, "under_test")
    guards.fired("event_test", 0, 3)
    snapshot = guards.snapshot()
    assert snapshot["over_test"] == [2, 6, 7.0]
    assert snapshot["under_test"] == [2, 4, 0.001]
    assert snapshot["event_test"][:2] == [0, 3] and np.isnan(snapshot["event_test"][2])
    guards.reset()
    assert guards.snapshot() == {}


def test_clipping_and_flooring_are_identity_inside_bounds():
    values = np.array([-7.0, -5.0, 0.3, 4.9, np.nan])
    np.testing.assert_array_equal(guards.clip_abs(values, 5.0, "clip_test"), [-5.0, -5.0, 0.3, 4.9, np.nan])
    np.testing.assert_array_equal(guards.floor(np.array([0.01, 0.05, 0.2]), 0.05, "floor_test"), [0.05, 0.05, 0.2])
    assert guards.snapshot()["clip_test"] == [1, 5, 7.0]
    assert guards.snapshot()["floor_test"] == [1, 3, 0.01]


def test_exponentiation_applies_configured_and_absolute_limits():
    values = np.array([14.0, -9.5, -60.0, 2000.0])
    bound = LIMITS.log2_max
    np.testing.assert_array_equal(exp2(values, bound), [2.0**bound, 2.0**-9.5, 2.0**-bound, 2.0**bound])
    snapshot = guards.snapshot()
    assert snapshot["log2_max"][:2] == [3, 4]
    assert snapshot["log2_overflow"][:2] == [1, 4]
    guards.reset()
    np.testing.assert_array_equal(exp2(values, None), [2.0**14, 2.0**-9.5, 2.0**-60, 2.0**1000])
    assert "log2_max" not in guards.snapshot()


def test_similarity_weights_floor_narrow_spread_and_clip_outliers():
    correlations = np.array([0.40, 0.50, 0.60])
    np.testing.assert_array_equal(
        sim_weight(correlations, 1.0, LIMITS.sim_sd_min, LIMITS.sim_z_clip),
        np.exp((correlations - correlations.mean()) / correlations.std()),
    )
    correlations = np.array([0.50, 0.51, 0.52])
    np.testing.assert_allclose(
        sim_weight(correlations, 1.0, LIMITS.sim_sd_min, LIMITS.sim_z_clip),
        np.exp((correlations - correlations.mean()) / LIMITS.sim_sd_min),
    )
    correlations = np.r_[np.zeros(99), 1.0]
    weights = sim_weight(correlations, 1.0, LIMITS.sim_sd_min, LIMITS.sim_z_clip)
    assert weights[-1] == pytest.approx(np.exp(LIMITS.sim_z_clip))
    assert guards.snapshot()["sim_z_clip"][:2] == [1, 106]


def test_noise_ratio_is_clipped_to_the_requested_bound(world):
    features = make_features(world)
    original = features.noise_ratio(1.0, LIMITS.nr_clip)
    guards.reset()
    hits = int(((original < 1 / 1.5) | (original > 1.5)).sum())
    np.testing.assert_array_equal(features.noise_ratio(1.0, 1.5), np.clip(original, 1 / 1.5, 1.5))
    assert hits > 0 and guards.snapshot()["nr_clip"][:2] == [hits, original.size]


def test_relative_variance_floor_handles_a_collapsed_estimate():
    effects = [np.zeros((1, 1))] * 3
    tables = [table(genes=["gene_a"], variance=[value]) for value in (1.0, 0.5, 1e-4)]
    result = inv_mad_dl(tables, ["gene_a"], lambda index: (effects[index], effects[index]), 3, 1, LIMITS.v_floor_rel)
    np.testing.assert_allclose(result[:, 0], [0.5, 1.0, 1 / LIMITS.v_floor_rel])
    assert guards.snapshot()["v_floor_rel"][:2] == [1, 3]


def test_similarity_softmax_uses_theorder_spread_floor():
    genes = [f"gene_{index}" for index in range(50)]
    base = np.random.default_rng(0).uniform(1, 100, 50)

    def controls(epsilon):
        return {
            line: (genes, base * np.exp(value * np.random.default_rng(index).normal(0, 1, 50)))
            for index, (line, value) in enumerate(zip(["cell_line_A", "cell_line_B", "cell_line_C"], epsilon))
        }

    for epsilon, hits in (((0.01, 0.02, 0.03), 1), ((0.05, 1.0, 3.0), 0)):
        guards.reset()
        weights, correlations = plan.similarity_weights(
            genes, base, controls(epsilon), ["cell_line_A", "cell_line_B", "cell_line_C"], 0.5, 50, LIMITS.order_sd_min
        )
        standardized = (correlations - correlations.mean()) / max(correlations.std(), LIMITS.order_sd_min)
        expected = np.exp((standardized - standardized.max()) / 0.5)
        np.testing.assert_allclose(weights, expected / expected.sum())
        assert guards.snapshot()["order_sd_min"][:2] == [hits, 1]


def test_placement_targets_are_clipped(monkeypatch):
    effects = np.array([-9.0, -1.0, 0.5, 7.4, 8.0])
    monkeypatch.setattr(placement, "dn_target", lambda *args: effects)
    monkeypatch.setattr(placement, "plan_sign", lambda values, *args: values)
    model = load_model(SOURCE_CONFIG)
    call = dict(priority=[1.0], idx=[0], sign=[1.0], nudge_strength=np.zeros(5), nudge_sign=np.zeros(5))
    targets = SimpleNamespace(cpm=np.ones(5), elig=np.ones(5, bool), tidx=np.array([-1]), lib=None)
    result = placement.job(0, "target_a", call, targets, np.ones((1, 5)), model)
    bound = model.guards.mt_ls_clip
    np.testing.assert_array_equal(result["mt"], 2.0 ** np.array([-bound, -1.0, 0.5, 7.4, bound]))
    assert guards.snapshot()["mt_ls_clip"][:2] == [2, 5]


def test_edge_events_are_counted():
    plan.cap_weights(np.ones(2), 0.3)
    placement.library_profile(np.array([0.0, 0.0, 100.0, 200.0]), 4000, 1.0)
    snapshot = guards.snapshot()
    assert snapshot["cap_weights_all_capped"][:2] == [1, 1]
    assert snapshot["empty_control_cells"][:2] == [2, 4]


def test_similarity_share_and_effective_count_aggregate_duplicate_rows():
    lines = [f"cell_line_{index:02d}" for index in range(40)]
    guards.share(lines, np.r_[1000.0, np.ones(39)], "pooling", LIMITS)
    snapshot = guards.snapshot()
    assert snapshot["sim_share_pooling"][0] == 1
    assert snapshot["sim_neff_pooling"][0] == 1
    guards.reset()
    guards.share(lines + lines, np.ones(80), "pooling", LIMITS)
    snapshot = guards.snapshot()
    assert snapshot["sim_share_pooling"][:2] == [0, 1]
    assert snapshot["sim_neff_pooling"][:2] == [0, 1]
    assert snapshot["sim_neff_pooling"][2] == pytest.approx(40)
