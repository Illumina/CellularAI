"""The CellularAI command-line interface."""

from __future__ import annotations

import sys

COMMANDS = {
    "run": ("cellularai.pipeline", "run a complete prediction"),
    "preflight": ("cellularai.preflight", "validate data and configuration"),
    "pool": ("cellularai.pooling", "pool summary effects for one context"),
    "predict": ("cellularai.prediction", "predict cells for one context"),
    "assemble": ("cellularai.assemble", "validate and merge predictions"),
    "evaluate": ("cellularai.evaluation", "stage or score an evaluation dataset"),
}


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: cellularai COMMAND [options]\n")
        for name, (_, description) in COMMANDS.items():
            print(f"  {name:<12} {description}")
        return
    command = argv.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}; choose from {', '.join(COMMANDS)}")
    module_name = COMMANDS[command][0]
    if module_name == "cellularai.evaluation":
        from .evaluation.run import main as entry
    else:
        from importlib import import_module

        entry = import_module(module_name).main
    entry(argv)


if __name__ == "__main__":
    main()
