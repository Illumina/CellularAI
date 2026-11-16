"""Validate context predictions and merge them into an h5ad prediction file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from . import guards
from .config import Config
from .evaluation.de import ControlOrderStats, de_table
from .task import NCELL, Task


def predicted_part(builds: Path, context: str) -> Path:
    return builds / f"ctx_{context}" / f"predicted_cells_{context}.h5ad"


def build_gates(report, limits, expected_provenance: dict) -> dict:
    """Apply build-report, numerical-guard and configuration-identity checks."""
    tally = {key: [int(value[0]), int(value[1]), float(value[2])] for key, value in report["guards"].items()}
    unmeasured_report = report["unmeasured"]
    unmeasured = sorted({str(target) for target in (*unmeasured_report["no_row"], *unmeasured_report["no_evidence"])})
    credit_share = float(report["credit_share"])
    readout = report.get("readout", {})
    missing = [str(value) for value in readout.get("missing", [])]
    near = [str(value) for value in readout.get("near_miss", [])]
    missing_fraction = len(missing) / int(readout["n_genes"]) if "n_genes" in readout else float("nan")
    provenance = report.get("provenance", {})
    absent = sorted(set(guards.NAMES) - set(tally))
    stale = sorted(set(limits.max_hit_rate) - set(guards.NAMES))
    checks = {
        "unmeasured": len(unmeasured) <= limits.max_unmeasured,
        "margin_reached": bool(report["margin_reached"]),
        "converged": bool(report["converged"]),
        "credit_share": credit_share < limits.credit_kink,
        "credit_band": credit_share >= limits.credit_band_lo,
        "readout_missing": bool(missing_fraction <= limits.max_missing_readout_frac),
        "provenance": all(str(provenance.get(key)) == value for key, value in expected_provenance.items()),
        "guards_registered": not absent and not stale,
    }
    failures = [f"{len(unmeasured)} unmeasured targets exceed the limit"] * (not checks["unmeasured"])
    failures += [f"{key} is false" for key in ("margin_reached", "converged") if not checks[key]]
    if not (checks["credit_share"] and checks["credit_band"]):
        failures.append(f"credit share {credit_share:.6f} is outside the accepted interval")
    if not checks["readout_missing"]:
        failures.append(f"missing-readout fraction {missing_fraction:.4g} exceeds the limit")
    if not checks["provenance"]:
        failures.append("prediction was built with different configuration or task inputs")
    if not checks["guards_registered"]:
        failures.append(f"guard registration mismatch: absent={absent}, unknown={stale}")
    for name, (hits, count, extreme) in tally.items():
        rate = limits.max_hit_rate.get(name, 0.0)
        checks[f"guard_{name}"] = hits <= rate * count
        if hits > rate * count:
            failures.append(f"guard {name}: {hits} of {count} values exceeded its bound (extreme {extreme:.3g})")
    return {
        **checks,
        "guard_tally": tally,
        "unmeasured_perturbations": unmeasured,
        "credit_share_value": credit_share,
        "readout_missing_genes": missing,
        "readout_missing_frac": missing_fraction,
        "readout_near_miss": near,
        "provenance_value": {key: str(value) for key, value in provenance.items()},
        "failures": failures,
    }


def gate(
    prediction: ad.AnnData,
    context: str,
    genes,
    perts,
    controls,
    call_depth,
    limits,
    expected_provenance,
    n_cells: int = NCELL,
) -> dict:
    targets = prediction.obs["target_gene"].astype(str).to_numpy()
    names, counts = np.unique(targets, return_counts=True)
    values = np.asarray(prediction.X.data)
    checks = {
        "genes": list(prediction.var_names) == list(genes),
        "context": set(prediction.obs["context"].astype(str)) == {context},
        "perturbations": sorted(names) == sorted(perts),
        "cells_per_perturbation": bool((counts == n_cells).all()),
        "sparse_csr": bool(sp.issparse(prediction.X) and prediction.X.getformat() == "csr"),
        "float32_counts": bool(prediction.X.dtype == np.float32),
        "integer_counts": bool(
            np.isfinite(values).all()
            and (values >= 0).all()
            and (values == np.round(values)).all()
            and (values < 2**24).all()
        ),
    }
    stats = ControlOrderStats.from_control_counts(controls.toarray())
    gene_positions = {gene: i for i, gene in enumerate(genes)}
    de_calls = []
    for target in perts:
        block = prediction.X[np.flatnonzero(targets == target)]
        block = block.toarray() if sp.issparse(block) else np.asarray(block)
        significant = de_table(block, stats)["significant"]
        if target in gene_positions:
            significant[gene_positions[target]] = False
        de_calls.append(int(significant.sum()))
    checks["median_n_pred"] = float(np.median(de_calls))
    checks["call_depth"] = checks["median_n_pred"] <= call_depth
    return {**checks, **build_gates(prediction.uns["build"], limits, expected_provenance)}


def merge_predictions(builds: Path, parts: list) -> None:
    """Atomically write the merged prediction h5ad after all context gates pass."""
    merged = builds / "prediction.h5ad"
    candidate = builds / "prediction.candidate.h5ad"
    ad.concat(parts, axis=0, join="inner").write_h5ad(candidate)
    os.replace(candidate, merged)
    print(f"[assembly] wrote {merged}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cellularai.assemble")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--builds", required=True)
    args = parser.parse_args(argv)
    builds = Path(args.builds)
    occupied = [name for name in ("prediction.h5ad",) if (builds / name).exists()]
    if occupied:
        raise SystemExit(
            f"[assembly] refusing to overwrite existing final artifacts {occupied}; use a fresh output directory"
        )
    config = Config.load(args.config)
    task = Task.load(args.task, config.training_data)
    expected = {"config": config.fingerprint, "task": task.fingerprint}
    parts, all_gates = [], {}
    for context in task.contexts:
        part = ad.read_h5ad(predicted_part(builds, context))
        all_gates[context] = gate(
            part,
            context,
            task.genes,
            task.perts,
            task.controls(context),
            task.cfg["call_depth"],
            config.model.guards,
            expected,
            task.n_cells,
        )
        print(f"[assembly] {context}: {all_gates[context]}", flush=True)
        parts.append(part)
    failed = [
        f"{context}.{key}"
        for context, values in all_gates.items()
        for key, value in values.items()
        if isinstance(value, (bool, np.bool_)) and not value
    ]
    (builds / "gates.json").write_text(json.dumps({"failed": failed, **all_gates}, indent=1) + "\n")
    if failed:
        details = [f"  {context}: {values['failures']}" for context, values in all_gates.items() if values["failures"]]
        raise SystemExit("\n".join(["[assembly] failed quality gates", *details]))
    merge_predictions(builds, parts)


if __name__ == "__main__":
    main()
