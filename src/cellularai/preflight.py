"""Validate configuration, summary payloads and task inputs before a long build."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .evaluation.data import DATASETS, dataset, load_inputs, reference_files
from .io import file_sha256, load_yaml
from .pooling.catalog import Catalog
from .task import Task


def verify_summary_bundle(catalog_path: Path) -> None:
    catalog_path = catalog_path.resolve()
    root = catalog_path.parent
    checksums = root / "SHA256SUMS"
    if not checksums.exists():
        raise FileNotFoundError(f"{root}: missing SHA256SUMS")
    expected = {}
    for line in checksums.read_text().splitlines():
        if not line.strip():
            continue
        digest_value, relative = line.split(None, 1)
        relative = relative.strip()
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("summary checksum manifest contains a path outside the bundle")
        expected[relative] = (digest_value, path)
    if not expected:
        raise ValueError("summary checksum manifest is empty")
    for relative, (wanted, path) in expected.items():
        if not path.is_file() or file_sha256(path) != wanted:
            raise ValueError(f"summary payload failed integrity verification: {relative}")
    catalog = Catalog.load(catalog_path)
    referenced = {"catalog.json", "SHA256SUMS", catalog.variance}
    for row in catalog.rows:
        referenced.update(row.files.values())
        if row.offset:
            referenced.add(row.offset)
    missing = sorted(referenced - set(expected) - {"catalog.json", "SHA256SUMS"})
    if missing:
        raise ValueError(f"catalog files missing from checksum manifest: {missing}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="cellularai preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--evaluation-config")
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    verify_summary_bundle(Path(config.training_data["catalog"]))
    task = Task.load(args.task, config.training_data)
    for context in task.contexts:
        task.controls(context)
    if args.evaluation_config:
        if not args.dataset:
            parser.error("--dataset is required with --evaluation-config")
        selected = dataset(args.dataset)
        files = reference_files(load_yaml(args.evaluation_config), selected)
        load_inputs(selected, files)
    print("Preflight passed: summary payloads, task controls and requested evaluation inputs are valid.")


if __name__ == "__main__":
    main()
