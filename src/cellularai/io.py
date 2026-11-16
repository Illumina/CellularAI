"""Small, safe file-reading and integrity helpers shared by the pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if duplicates := sorted({str(key) for key in keys if keys.count(key) > 1}):
            raise ValueError(f"duplicate YAML keys {duplicates} {node.start_mark}")
        return super().construct_mapping(node, deep)


def load_yaml(path) -> dict:
    return yaml.load(Path(path).read_text(), Loader=_UniqueKeyLoader)


def digest(value) -> str:
    """SHA-256 of canonical JSON for a YAML/JSON-like value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha256(path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            value.update(chunk)
    return value.hexdigest()


def code_version() -> dict:
    """Identify public code by package version and a path-independent source-tree hash."""
    from . import __version__

    root = Path(__file__).resolve().parent
    value = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        value.update(f"{path.relative_to(root)}\n".encode() + path.read_bytes())
    return {"package_version": __version__, "code_sha256": value.hexdigest()}


def md5(path) -> str:
    value = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            value.update(chunk)
    return value.hexdigest()


def h5_strings(node) -> list[str]:
    decode = lambda value: value.decode() if isinstance(value, bytes) else str(value)  # noqa: E731
    if isinstance(node, h5py.Group):
        categories = [decode(value) for value in node["categories"][:]]
        return [categories[index] for index in node["codes"][:]]
    return [decode(value) for value in node[:]]


def read_column(path, column) -> list[str]:
    """Read literal strings so gene names such as 'NA' never become missing values."""
    with Path(path).open(newline="") as handle:
        return [row[column].strip() for row in csv.DictReader(handle)]


def write_column(path, column, values) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([column])
        writer.writerows([[value] for value in values])
