"""Load a prediction task and its context-specific control counts.

A task defines control files, the ordered readout genes, target perturbations and generation limits. Its content fingerprint hashes the actual task inputs rather than their installation paths.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from .config import Config
from .io import digest, load_yaml, read_column
from .pooling.catalog import check_exclude

NCELL = 400  # default cells per perturbation; override with the task key cells_per_perturbation
KEYS = {
    "data_root",
    "contexts",
    "perturbations",
    "readout_genes",
    "call_depth",
    "median_degs",
    "exclude_cell_lines",
    "control_label",
}
OPTIONAL = {"round", "extra", "cells_per_perturbation", "seed"}


def check_cfg(cfg, where) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"{where}: not a mapping")
    if (KEYS - set(cfg)) or (set(cfg) - KEYS - OPTIONAL):
        raise ValueError(f"{where}: missing {sorted(KEYS - set(cfg))}, unknown {sorted(set(cfg) - KEYS - OPTIONAL)}")
    contexts = cfg["contexts"]
    if not isinstance(contexts, dict) or not contexts:
        raise ValueError(f"{where}: contexts must be a non-empty mapping")
    for context, value in contexts.items():
        if not isinstance(context, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", context) is None:
            raise ValueError(f"{where}: invalid context name {context!r}")
        if not isinstance(value, dict) or set(value) != {"controls"} or not isinstance(value["controls"], str):
            raise ValueError(f"{where}: context {context}: expected exactly {{controls: <path>}}")
    for key in ("call_depth", "median_degs"):
        if (
            isinstance(cfg[key], bool)
            or not isinstance(cfg[key], (int, float))
            or not np.isfinite(cfg[key])
            or cfg[key] <= 0
        ):
            raise ValueError(f"{where}: {key} must be a positive number")
    if "cells_per_perturbation" in cfg:
        n = cfg["cells_per_perturbation"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ValueError(f"{where}: cells_per_perturbation must be an integer >= 2")
    if "seed" in cfg:
        seed = cfg["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{where}: seed must be a non-negative integer")
    excluded = cfg["exclude_cell_lines"]
    if not isinstance(excluded, list) or not all(isinstance(item, str) and item for item in excluded):
        raise ValueError(f"{where}: exclude_cell_lines must be a list of names")
    if not isinstance(cfg["control_label"], str) or not cfg["control_label"].strip():
        raise ValueError(f"{where}: control_label must be a non-empty string")


def check_names(names, what: str) -> None:
    if blank := sum(not name for name in names):
        raise ValueError(f"{what}: {blank} empty name(s)")
    if duplicates := sorted(name for name, count in Counter(names).items() if count > 1):
        raise ValueError(f"{what}: duplicate names {duplicates[:10]}")


def _file_sha256(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


@dataclass(frozen=True)
class Task:
    path: Path
    cfg: dict
    genes: list[str]
    perts: list[str]
    identity: dict

    @classmethod
    def load(cls, path, training_data: dict) -> "Task":
        path = Path(path).resolve()
        cfg = load_yaml(path)
        check_cfg(cfg, path)
        root = Path(cfg["data_root"])
        if not root.is_absolute():
            root = path.parent / root
        root = root.resolve()
        cfg["data_root"] = str(root)
        genes = read_column(root / cfg["readout_genes"], "gene_name")
        check_names(genes, cfg["readout_genes"])
        label = cfg["control_label"]
        perts = sorted(target for target in read_column(root / cfg["perturbations"], "target_gene") if target != label)
        if len(set(perts)) != len(perts):
            raise ValueError(f"{cfg['perturbations']}: duplicate perturbations")
        check_names(perts, cfg["perturbations"])
        if similar := [target for target in perts if target.lower() == label.lower()]:
            raise ValueError(f"{cfg['perturbations']}: {similar} differ from control_label only by case")
        if len(perts) < 3:
            raise ValueError(f"{cfg['perturbations']}: at least three perturbations are required")
        check_exclude(training_data, cfg["exclude_cell_lines"])
        identity = {
            key: value
            for key, value in cfg.items()
            if key not in {"data_root", "contexts", "perturbations", "readout_genes"}
        }
        identity.update(
            {
                "gene_file_sha256": _file_sha256(root / cfg["readout_genes"]),
                "perturbation_file_sha256": _file_sha256(root / cfg["perturbations"]),
                "genes": genes,
                "perts": perts,
                "controls": {
                    context: _file_sha256(root / value["controls"]) for context, value in cfg["contexts"].items()
                },
            }
        )
        return cls(path, cfg, genes, perts, identity)

    @property
    def fingerprint(self) -> str:
        return digest(self.identity)

    @property
    def n_cells(self) -> int:
        """Cells generated per perturbation per context."""
        return int(self.cfg.get("cells_per_perturbation", NCELL))

    @property
    def seed(self) -> int | None:
        """Synthesis seed mixed into each perturbation's generator; None keeps the target-name-only streams."""
        return self.cfg.get("seed")

    @property
    def root(self) -> Path:
        return Path(self.cfg["data_root"])

    @property
    def contexts(self) -> list[str]:
        return list(self.cfg["contexts"])

    @property
    def exclude(self) -> list[str]:
        return self.cfg["exclude_cell_lines"]

    def controls(self, context: str) -> sp.csr_matrix:
        if context not in self.cfg["contexts"]:
            raise ValueError(f"{context!r} is not a task context")
        var, counts = read_controls(self.root / self.cfg["contexts"][context]["controls"])
        if var != self.genes:
            raise ValueError(f"context {context} control genes differ from {self.cfg['readout_genes']}")
        if counts.shape[0] < 2:
            raise ValueError(f"context {context}: at least two control cells are required")
        if empty := int((np.asarray(counts.sum(axis=1)).ravel() == 0).sum()):
            raise ValueError(f"context {context}: {empty} control cells have no counts")
        return counts


def stage_args(prog: str, argv, *required: str, optional: tuple = ()):
    parser = argparse.ArgumentParser(prog=prog)
    for flag in ("--config", "--task", *required):
        parser.add_argument(flag, required=True)
    for flag, kwargs in optional:
        parser.add_argument(flag, **kwargs)
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    return args, config, Task.load(args.task, config.training_data)


def read_controls(path) -> tuple[list[str], sp.csr_matrix]:
    """Read dense/CSR/CSC control counts without materializing a full AnnData object."""
    with h5py.File(path, "r") as handle:
        var = handle["var"]
        idx = var[var.attrs["_index"]]
        if isinstance(idx, h5py.Group) and "values" in idx:
            if idx["mask"][:].any():
                raise ValueError(f"{path}: missing gene names")
            idx = idx["values"]
        elif isinstance(idx, h5py.Group):
            codes = idx["codes"][:]
            if (codes < 0).any():
                raise ValueError(f"{path}: missing gene names")
            idx = idx["categories"][:][codes]
        names = [value.decode() if isinstance(value, bytes) else str(value) for value in idx[:]]
        matrix = handle["X"]
        if isinstance(matrix, h5py.Group):
            shape = tuple(int(value) for value in matrix.attrs["shape"])
            encoding = matrix.attrs["encoding-type"]
            encoding = encoding.decode() if isinstance(encoding, bytes) else str(encoding)
            if encoding not in ("csr_matrix", "csc_matrix"):
                raise ValueError(f"{path}: X must be CSR or CSC")
            result = (sp.csr_matrix if encoding == "csr_matrix" else sp.csc_matrix)(
                (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=shape
            )
            result.check_format(full_check=True)
        else:
            result = matrix[:]
    if result.ndim != 2 or result.shape[1] != len(names):
        raise ValueError(f"{path}: matrix and gene-axis dimensions differ")
    result = sp.csr_matrix(result).astype(np.float64)
    result.sum_duplicates()
    check_counts(result.data, path)
    return names, result


def check_counts(values, what) -> None:
    values = np.asarray(values)
    if not (np.isfinite(values) & (values >= 0) & (values == np.round(values)) & (values < 2**24)).all():
        raise ValueError(f"{what}: control counts must be finite non-negative integers below 2^24")


def control_cpm_mean(counts) -> np.ndarray:
    """Compute per-gene mean cell-level CPM in bounded memory chunks."""
    accumulator = np.zeros(counts.shape[1], dtype=np.float64)
    counts = sp.csr_matrix(counts)
    for start in range(0, counts.shape[0], 2048):
        dense = np.asarray(counts[start : start + 2048].todense(), dtype=np.float64)
        totals = np.maximum(dense.sum(axis=1, keepdims=True), 1.0)
        accumulator += (1e6 * dense / totals).sum(axis=0)
    return accumulator / counts.shape[0]
