"""Stage an evaluation dataset as an ordinary CellularAI prediction task."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from ..io import write_column
from ..task import check_counts
from .data import Inputs


def stage(inputs: Inputs, task_template: dict, out: Path) -> Path:
    counts = np.asarray(inputs.controls, dtype=np.float64)
    check_counts(counts, inputs.dataset)
    out.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=sp.csr_matrix(counts.astype(np.float32)), var=pd.DataFrame(index=inputs.genes)).write_h5ad(
        out / "controls.h5ad"
    )
    write_column(out / "gene_names.csv", "gene_name", inputs.genes)
    write_column(out / "pert_counts.csv", "target_gene", inputs.perts)
    task = {
        "round": inputs.dataset,
        "data_root": str(out),
        "contexts": {inputs.dataset: {"controls": "controls.h5ad"}},
        "perturbations": "pert_counts.csv",
        "readout_genes": "gene_names.csv",
        "call_depth": task_template["call_depth"],
        "median_degs": task_template["median_degs"],
        "exclude_cell_lines": [],
        "control_label": task_template["control_label"],
        "extra": task_template.get("extra", {}),
    }
    for key in ("cells_per_perturbation", "seed"):
        if key in task_template:
            task[key] = task_template[key]
    (out / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False))
    print(
        f"[stage] {inputs.dataset}: {counts.shape[0]} controls, {len(inputs.genes)} genes, "
        f"{len(inputs.perts)} perturbations -> {out}",
        flush=True,
    )
    return out / "task.yaml"
