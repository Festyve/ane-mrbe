#!/usr/bin/env python3
"""Run RQ1 (role-decodability linear probes) and print the JSON summary.

Usage: run_rq1.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]

Caches no-edit activations for every primary-stimulus cell, probes each
(injection site x activation source) with the leave-one-pair-out ridge probe,
and writes rq1_probe.json + the selectivity figure. --dry-run uses the
DummyModel's planted ground truth: binding mode puts the role signal in the
J-space component, bag mode in the orthogonal remainder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jspace_binding.config import Config
from jspace_binding.experiments.rq1_probe import run_rq1
from jspace_binding.model.base import ProbeActivationSource
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.generate import generate_families


def _build_model(config: Config, dry_run: bool, dummy_mode: str | None) -> ProbeActivationSource:
    backend = "dummy" if dry_run else config.model.backend
    if backend == "dummy":
        return DummyModel(mode=dummy_mode or config.model.dummy_mode, seed=config.experiment.seed)
    if backend == "qwen_jlens":
        from jspace_binding.model.qwen_jlens import QwenJLensModel

        return QwenJLensModel(config.model, directions_dir=config.paths.directions)
    sys.exit(f"unknown model.backend {config.model.backend!r}; expected 'dummy' or 'qwen_jlens'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RQ1 linear-probe analysis.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--dry-run", action="store_true", help="force the GPU-free DummyModel backend"
    )
    parser.add_argument("--dummy-mode", choices=("binding", "bag"), default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = _build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    families = generate_families(config)
    print(f"generated {len(families)} families", file=sys.stderr)
    try:
        summary = run_rq1(config, model, families)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"backend {config.model.backend!r} cannot run yet: {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
