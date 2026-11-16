"""Run the shared prediction/assembly pipeline locally or emit equivalent Slurm jobs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .io import load_yaml
from .task import check_cfg


def commands(
    config: Path,
    task: Path,
    out: Path,
    python: str,
    workers: int | None = None,
    score: str | None = None,
    evaluation_config: Path | None = None,
) -> dict[str, list[list[str]]]:
    common = ["--config", str(config), "--task", str(task)]
    jobs: dict[str, list[list[str]]] = {}
    for context in load_yaml(task)["contexts"]:
        directory = out / f"ctx_{context}"
        pooled = directory / f"pooled_effects_{context}.npz"
        predicted = directory / f"predicted_cells_{context}.h5ad"
        prediction = [
            python,
            "-u",
            "-m",
            "cellularai.prediction",
            *common,
            "--context",
            context,
            "--pooling",
            str(pooled),
            "--out",
            str(predicted),
        ]
        if workers is not None:
            prediction.extend(["--n-jobs", str(workers)])
        jobs[f"ctx_{context}"] = [
            [python, "-u", "-m", "cellularai.pooling", *common, "--context", context, "--out", str(pooled)],
            prediction,
        ]
    assembly = [[python, "-u", "-m", "cellularai.assemble", *common, "--builds", str(out)]]
    if score:
        if evaluation_config is None:
            raise ValueError("--evaluation-config is required with --score")
        assembly.append(
            [
                python,
                "-u",
                "-m",
                "cellularai.evaluation",
                "score",
                "--dataset",
                score,
                "--config",
                str(evaluation_config),
                "--prediction",
                str(out / "prediction.h5ad"),
                "--name",
                out.name,
                "--out",
                str(out / "score"),
            ]
        )
    jobs["assemble"] = assembly
    return jobs


def _environment() -> dict[str, str]:
    return {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def run_local(jobs: dict[str, list[list[str]]], out: Path, compute: dict) -> None:
    temporary = Path(compute.get("tmp", out / "tmp"))
    temporary.mkdir(parents=True, exist_ok=True)
    environment = {**_environment(), "TMPDIR": str(temporary), "XDG_CACHE_HOME": str(temporary / "cache")}
    for name, stage_commands in jobs.items():
        for command in stage_commands:
            print(f"[pipeline] {name}: {shlex.join(command)}", flush=True)
            subprocess.run(command, check=True, env=environment)


def write_slurm(jobs: dict[str, list[list[str]]], out: Path, compute: dict) -> Path:
    """Write dependency-chained jobs; nothing is submitted automatically."""
    scheduler = compute["slurm"]
    temporary = Path(compute.get("tmp", out / "tmp"))
    temporary.mkdir(parents=True, exist_ok=True)
    directory = out / "jobs"
    directory.mkdir(parents=True, exist_ok=True)
    for name, stage_commands in jobs.items():
        script = [
            "#!/bin/bash",
            f"#SBATCH -J {shlex.quote(out.name + '_' + name)}",
            f"#SBATCH -o {shlex.quote(str(directory / (name + '_%j.log')))}",
            "set -euo pipefail",
            "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1",
            f"export TMPDIR={shlex.quote(str(temporary))} XDG_CACHE_HOME={shlex.quote(str(temporary / 'cache'))}",
            *(shlex.join(command) for command in stage_commands),
            "",
        ]
        (directory / f"{name}.sh").write_text("\n".join(script))
    contexts = [name for name in jobs if name != "assemble"]
    identifiers = {name: f"job_{index}" for index, name in enumerate(contexts)}
    launch = ["#!/bin/bash", "set -euo pipefail"]
    launch += [
        f"{identifiers[name]}=$(sbatch --parsable {scheduler['build']} {shlex.quote(str(directory / (name + '.sh')))})"
        for name in contexts
    ]
    launch.append(
        f"sbatch {scheduler['assemble']} --dependency=afterok:"
        + ":".join(f"${identifiers[name]}" for name in contexts)
        + f" {shlex.quote(str(directory / 'assemble.sh'))}"
    )
    (directory / "launch.sh").write_text("\n".join(launch) + "\n")
    return directory / "launch.sh"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cellularai.pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--score", choices=("vcc2025",))
    parser.add_argument("--evaluation-config")
    parser.add_argument("--scheduler", choices=("local", "slurm"))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    task_path = Path(args.task).resolve()
    check_cfg(load_yaml(task_path), task_path)
    compute = load_yaml(config_path).get("compute", {})
    python = compute.get("python", sys.executable)
    workers = args.workers if args.workers is not None else compute.get("workers")
    jobs = commands(
        config_path,
        task_path,
        Path(args.out).resolve(),
        python,
        workers,
        args.score,
        Path(args.evaluation_config).resolve() if args.evaluation_config else None,
    )
    scheduler = args.scheduler or compute.get("backend", "local")
    if scheduler == "slurm":
        print(write_slurm(jobs, Path(args.out).resolve(), compute))
    else:
        run_local(jobs, Path(args.out).resolve(), compute)


if __name__ == "__main__":
    main()
