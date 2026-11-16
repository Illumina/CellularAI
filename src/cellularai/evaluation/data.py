"""Resolve and integrity-check the supported evaluation data bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import file_sha256, md5


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    data: str


_DATASETS = (
    EvaluationDataset("vcc2025", "vcc2025"),
)
DATASETS = {dataset.name: dataset for dataset in _DATASETS}


def dataset(name: str) -> EvaluationDataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(f"unknown evaluation dataset {name!r}; choose from {sorted(DATASETS)}") from None


FILE_KEYS = ("controls", "perturbations", "provision", "anchor_scores", "reference_side")


def reference_files(config: dict, selected: EvaluationDataset) -> dict[str, Path]:
    entry = config["datasets"][selected.data]
    root = Path(entry["root"])
    files = {key: root / entry["files"][key] for key in FILE_KEYS}
    for key, expected in entry.get("md5", {}).items():
        actual = md5(files[key])
        if actual != expected:
            raise ValueError(f"{selected.data}: {key} MD5 does not match the data release")
    for key, expected in entry.get("sha256", {}).items():
        actual = file_sha256(files[key])
        if actual != expected:
            raise ValueError(f"{selected.data}: {key} SHA-256 does not match the data release")
    return files


@dataclass(frozen=True)
class Inputs:
    dataset: str
    genes: list[str]
    perts: list[str]
    controls: np.ndarray


def load_inputs(selected: EvaluationDataset, files: dict[str, Path]) -> Inputs:
    with np.load(files["controls"], allow_pickle=False) as archive:
        pool = archive["counts"]
        genes = [str(gene) for gene in archive["genes"]]
    indices = np.load(files["provision"], allow_pickle=False)
    if indices.size == 0 or indices.min() < 0 or indices.max() >= pool.shape[0]:
        raise ValueError(f"{selected.data}: control selection is empty or out of bounds")
    perturbations = json.loads(Path(files["perturbations"]).read_text())
    return Inputs(dataset=selected.name, genes=genes, perts=perturbations, controls=pool[indices])


ANCHOR_CODE_VERSION = "3-doc-fidelity+control-provision-18400"


def load_anchors(path) -> tuple[dict, dict]:
    payload = json.loads(Path(path).read_text())
    if payload.get("code_version") != ANCHOR_CODE_VERSION:
        raise ValueError(f"{path}: unsupported anchor formula version")

    def numeric(values):
        return {key: float("nan") if value is None else float(value) for key, value in values.items()}

    return numeric(payload["baseline"]), numeric(payload["replicate"])
