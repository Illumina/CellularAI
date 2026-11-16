"""Stage, score and compare predictions on the supported evaluation datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ..io import load_yaml
from ..task import NCELL
from .data import EvaluationDataset, Inputs, dataset, load_anchors, load_inputs, reference_files
from .scoring import METRIC_NAMES, SideStats, build_prediction_side, expr_mse_parts, scaled_scores, score_panel
from .stage import stage


def _json_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def panel_targets(genes: list[str], perts: list[str]) -> list[int]:
    positions = {gene: i for i, gene in enumerate(genes)}
    return [positions[target] for target in perts if target in positions]


def score_block(
    selected: EvaluationDataset, block: np.ndarray, inputs: Inputs, files: dict, method: str, n_jobs: int = 1
) -> dict:
    expected = (len(inputs.perts), block.shape[1] if block.ndim == 3 else NCELL, len(inputs.genes))
    if block.shape != expected or block.dtype != np.float32 or expected[1] < 2:
        raise ValueError(f"prediction has shape/dtype {block.shape}/{block.dtype}; expected {expected}/float32")
    baseline, replicate = load_anchors(files["anchor_scores"])
    reference = SideStats.load(files["reference_side"])
    if reference.genes != inputs.genes or reference.perts != inputs.perts:
        raise ValueError(f"{selected.name}: cached reference axes do not match the evaluation inputs")
    for target in reference.perts:
        if reference.profiles[target].shape != (len(inputs.genes),):
            raise ValueError(f"{selected.name}: cached reference profile has an incompatible shape")
        table = reference.de.get(target)
        if table is None or any(np.asarray(values).shape != (len(inputs.genes),) for values in table.values()):
            raise ValueError(f"{selected.name}: cached reference differential-expression arrays are incompatible")
    prediction = build_prediction_side(
        {target: block[i] for i, target in enumerate(inputs.perts)},
        inputs.controls,
        inputs.genes,
        inputs.perts,
        n_jobs=n_jobs,
    )
    raw = score_panel(prediction, reference, panel_targets(inputs.genes, inputs.perts))
    return {
        "method": method,
        "dataset": selected.name,
        "raw": {key: _json_float(value) for key, value in raw.items()},
        "scaled": {key: _json_float(value) for key, value in scaled_scores(raw, baseline, replicate).items()},
        "baseline": {key: _json_float(value) for key, value in baseline.items()},
        "replicate": {key: _json_float(value) for key, value in replicate.items()},
        "mse_parts": {key: _json_float(value) for key, value in expr_mse_parts(prediction, reference).items()},
    }


def block_from_h5ad(path, inputs: Inputs) -> np.ndarray:
    import anndata as ad

    prediction = ad.read_h5ad(path)
    if list(prediction.var_names) != inputs.genes:
        raise ValueError(f"{path}: gene axis differs from the evaluation dataset")
    targets = prediction.obs["target_gene"].astype(str).to_numpy()
    if set(targets) != set(inputs.perts):
        raise ValueError(f"{path}: perturbations differ from the evaluation dataset")
    sizes = {target: int((targets == target).sum()) for target in inputs.perts}
    n_cells = sizes[inputs.perts[0]]
    if unequal := {target: size for target, size in sizes.items() if size != n_cells}:
        raise ValueError(
            f"{path}: perturbations must have equal cell counts; {inputs.perts[0]} has {n_cells}, {unequal}"
        )
    if n_cells < 2:
        raise ValueError(f"{path}: at least two cells per perturbation are required")
    if n_cells != NCELL:
        print(
            f"[score] WARNING: {n_cells} cells per perturbation; reference anchors assume {NCELL}, "
            "so scaled scores are not directly comparable",
            flush=True,
        )
    block = np.empty((len(inputs.perts), n_cells, len(inputs.genes)), np.float32)
    for i, target in enumerate(inputs.perts):
        block[i] = prediction.X[np.flatnonzero(targets == target)].toarray()
    return block


def tables(arms: dict[str, str | Path]) -> str:
    scores = {arm: json.loads(Path(path).read_text()) for arm, path in arms.items()}
    width = max(len("arm"), *(len(arm) for arm in scores))
    lines = []
    for title, key, columns in (
        ("RAW", "raw", METRIC_NAMES),
        ("SCALED", "scaled", METRIC_NAMES + ("overall",)),
    ):
        lines += [title, f"{'arm':<{width}} " + " ".join(f"{column:>9}" for column in columns)]
        for arm, score in scores.items():
            values = (score[key].get(column) for column in columns)
            lines.append(
                f"{arm:<{width}} " + " ".join(f"{'nan':>9}" if value is None else f"{value:9.5f}" for value in values)
            )
        lines.append("")
    return "\n".join(lines)


def cmd_stage(args, config) -> None:
    selected = dataset(args.dataset)
    files = reference_files(config, selected)
    stage(
        load_inputs(selected, files),
        load_yaml(args.like),
        Path(args.out).resolve(),
    )


def cmd_score(args, config) -> None:
    selected = dataset(args.dataset)
    files = reference_files(config, selected)
    inputs = load_inputs(selected, files)
    payload = score_block(
        selected,
        block_from_h5ad(args.prediction, inputs),
        inputs,
        files,
        args.name or "prediction",
        args.n_jobs,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(payload, indent=1))
    print(f"[{selected.name}] {payload['method']} scaled overall={payload['scaled']['overall']:.5f}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cellularai.evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--dataset", required=True, choices=("vcc2025",))
    score.add_argument("--config", required=True)
    score.add_argument("--prediction", required=True)
    score.add_argument("--name")
    score.add_argument("--out", required=True)
    score.add_argument("--n-jobs", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--dataset", required=True, choices=("vcc2025",))
    stage_parser.add_argument("--config", required=True)
    stage_parser.add_argument("--like", required=True, help="task YAML supplying generation settings")
    stage_parser.add_argument("--out", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("arms", nargs="+", help="ARM=path/to/scores.json")
    args = parser.parse_args(argv)
    if args.command == "report":
        print(tables(dict(value.split("=", 1) for value in args.arms)), end="")
        return
    config = load_yaml(args.config)
    {"stage": cmd_stage, "score": cmd_score}[args.command](args, config)


if __name__ == "__main__":
    main()
