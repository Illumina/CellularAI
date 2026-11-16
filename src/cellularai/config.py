"""Validated model, data-location and execution configuration.

Every model section is a frozen dataclass. Missing and unknown keys fail immediately so a long prediction never silently uses an unintended default. Fingerprints cover the portable catalog and model values, not installation paths or compute settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass, make_dataclass
from pathlib import Path

from .io import digest, load_yaml


def section(name: str, keys: str, types: dict | None = None, **methods):
    types = types or {}
    cls = make_dataclass(name, [(key, types.get(key, object)) for key in keys.split()], frozen=True, namespace=methods)
    cls.__module__ = __name__
    return cls


def _pooling_check(self):
    if (self.correction, self.fill, self.one_stream) != ("LZ", "zf", True):
        raise ValueError("only correction 'LZ', fill 'zf' and one_stream true are supported")
    object.__setattr__(self, "views", tuple(self.views))


def _totals_check(self):
    if isinstance(self.fixed_steps, bool) or not isinstance(self.fixed_steps, int) or self.fixed_steps < 1:
        raise ValueError("totals.fixed_steps must be a positive integer")
    if isinstance(self.fixed_rtol, bool) or not isinstance(self.fixed_rtol, (int, float)) or not self.fixed_rtol > 0:
        raise ValueError("totals.fixed_rtol must be a positive number")


OPS = ("l2_row", "l2_col", "l2_mat", "z_row", "z_col", "z_mat")
Pooling = section(
    "Pooling",
    "views auxiliary_weight " + " ".join(OPS) + " noise se se_clip kd_ref kd_floor sim sim_hvg pair_weight "
    "correction fill one_stream",
    __post_init__=_pooling_check,
    ops=lambda self: {key: getattr(self, key) for key in OPS},
)
Calls = section(
    "Calls", "eligible_cpm min_lfc gamma n_min n_max w_pow phi strength_z strength_k strength_c strength_phi"
)
Plan = section(
    "Plan",
    "curve_slope sim_tau sim_ntop sim_cap plan_z_uniform xsim_tau xsim_strength xsim_cap_factor xsim_max_total "
    "power_k power_c power_phi power_depth power_z_n2 power_t_pen power_soft_z shrink_quantile shrink_factor",
)
Totals = section(
    "Totals",
    "credit_draws credit_seed amp_share shape_gamma shape_kz mix gate_z pcs_cells pcs_seed cp_j0 cp_k cp_lam cp_top "
    "pw_gamma pw_k pw_lam pw_top residual_rank residual_ridge residual_min_measured residual_alpha residual_perts "
    "residual_scale_floor residual_global residual_proj_ridge residual_clip fixed_steps fixed_rtol",
    __post_init__=_totals_check,
)
Rule = section("Rule", "amp h0 kz phi cap cl2")
Placement = section(
    "Placement",
    "rule frozen evidence_z rule_band boost_band boost damp_band damp plan_floor_z order_az order_zsim order_clip lib_spread",
    {"rule": Rule, "frozen": Rule},
)
Synthesis = section(
    "Synthesis",
    "z_plan z_nudge rmax n_bis ztol ztol_plan ltol ne_burst planned_z overshoot_z repair_r solve_ntop solve_tol "
    "solve_iters shift_band sparsify_step",
)
Guards = section(
    "Guards",
    "l2_abs_max z_abs_max v_floor_rel sim_sd_min sim_z_clip nr_clip log2_max mt_ls_clip order_sd_min sim_share_max "
    "sim_neff_min max_unmeasured max_missing_readout_frac credit_kink credit_band_lo max_hit_rate",
)
Model = section(
    "Model",
    "pooling calls plan totals placement synthesis guards",
    {
        "pooling": Pooling,
        "calls": Calls,
        "plan": Plan,
        "totals": Totals,
        "placement": Placement,
        "synthesis": Synthesis,
        "guards": Guards,
    },
)


def _make(cls, value, where: str):
    expected = {field.name for field in fields(cls)}
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{where}: unknown {sorted(actual - expected)}, missing {sorted(expected - actual)}")
    types = {field.name: field.type for field in fields(cls)}
    return cls(
        **{
            key: _make(types[key], item, f"{where}.{key}") if is_dataclass(types[key]) else item
            for key, item in value.items()
        }
    )


def load_model(path) -> Model:
    return _make(Model, load_yaml(path)["model"], "model")


@dataclass(frozen=True)
class Config:
    training_data: dict
    compute: dict
    model: Model
    fingerprint: str = ""

    @classmethod
    def load(cls, path) -> "Config":
        path = Path(path).resolve()
        payload = load_yaml(path)
        training_data = dict(payload["training_data"])
        catalog = Path(training_data["catalog"])
        if not catalog.is_absolute():
            catalog = path.parent / catalog
        training_data["catalog"] = str(catalog.resolve())
        identity = {"catalog": json.loads(catalog.read_text()), "model": payload["model"]}
        return cls(
            training_data=training_data,
            compute=payload.get("compute", {}),
            model=_make(Model, payload["model"], "model"),
            fingerprint=digest(identity),
        )
