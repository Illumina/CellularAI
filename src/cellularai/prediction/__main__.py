"""Generate integer-count cells (400 per perturbation by default) for one context."""

from __future__ import annotations

import time
from pathlib import Path

import anndata as ad
import pandas as pd

from ..io import code_version
from ..pooling.build import Evidence, training_rows
from ..pooling.catalog import Catalog, readout_coverage
from ..task import stage_args
from .build import build
from .inputs import GenerationInputs


def main(argv=None) -> None:
    start = time.monotonic()
    args, config, task = stage_args(
        "python -m cellularai.prediction",
        argv,
        "--context",
        "--pooling",
        "--out",
        optional=(("--n-jobs", dict(type=int)),),
    )
    controls = task.controls(args.context)
    genes, perts = task.genes, task.perts
    evidence = Evidence.load(args.pooling, genes, perts)
    inputs = GenerationInputs.load(
        config.training_data,
        genes,
        perts,
        config.model.totals.residual_perts,
        task.exclude,
    )
    print(f"[prediction] inputs loaded: {time.monotonic() - start:.1f}s", flush=True)
    rows = training_rows(Catalog.load(config.training_data), config.model.pooling, task.exclude)
    readout = readout_coverage(genes, rows)
    print(
        f"[prediction] readouts without training evidence: {len(readout['missing'])} of {len(genes)}; "
        f"near matches {readout['near_miss']}",
        flush=True,
    )
    matrix, report = build(
        controls,
        genes,
        perts,
        evidence,
        inputs,
        task.cfg["call_depth"],
        task.cfg["median_degs"],
        config.model,
        n_jobs=args.n_jobs if args.n_jobs is not None else int(config.compute.get("workers", 1)),
        n_cells=task.n_cells,
        seed=task.seed,
    )
    report["readout"] = readout
    report["provenance"] = {"config": config.fingerprint, "task": task.fingerprint, **code_version()}
    observations = pd.DataFrame(
        {
            "context": pd.Categorical([args.context] * (len(perts) * task.n_cells)),
            "target_gene": [target for target in perts for _ in range(task.n_cells)],
        }
    )
    observations.index = [f"{args.context}_{target}_{i}" for target in perts for i in range(task.n_cells)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=matrix, obs=observations, var=pd.DataFrame(index=genes), uns={"build": report}).write_h5ad(args.out)
    print(f"[prediction] context {args.context} -> {args.out}: {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
