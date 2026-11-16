"""Command-line entry point for context-specific effect pooling."""

from __future__ import annotations

import time

from ..task import control_cpm_mean, stage_args
from .build import build


def main(argv=None) -> None:
    start = time.monotonic()
    args, config, task = stage_args("python -m cellularai.pooling", argv, "--context", "--out")
    mean_cpm = control_cpm_mean(task.controls(args.context))
    build(
        config.training_data,
        config.model.pooling,
        config.model.guards,
        task.genes,
        task.perts,
        mean_cpm,
        task.exclude,
    ).save(args.out, task.genes, task.perts)
    print(f"[pooling] context {args.context} -> {args.out}: {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
