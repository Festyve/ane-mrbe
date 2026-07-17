#!/usr/bin/env python3
"""Run RQ2 (J-space ablation vs matched controls) and print the JSON summary.

Usage: run_rq2.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]

Scores the binding task (ROLE probe) and the difficulty-matched recall task
(NEUTRAL probe) under NO_EDIT / ABLATE_JSPACE / ABLATE_RANDOM_SUBSPACE, and
writes rq2_ablation.json + the ablation-deltas figure. --dry-run uses the
DummyModel's planted ground truth: binding mode shows a binding-specific
J-space deficit exceeding the random-subspace bar; bag mode does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jspace_binding.config import Config
from jspace_binding.experiments.rq2_ablation import run_rq2
from jspace_binding.model.base import WorkspaceModel
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.generate import generate_families


def _build_model(config: Config, dry_run: bool, dummy_mode: str | None) -> WorkspaceModel:
    backend = "dummy" if dry_run else config.model.backend
    if backend == "dummy":
        return DummyModel(mode=dummy_mode or config.model.dummy_mode, seed=config.experiment.seed)
    if backend == "qwen_jlens":
        from jspace_binding.model.qwen_jlens import QwenJLensModel

        return QwenJLensModel(config.model, directions_dir=config.paths.directions)
    sys.exit(f"unknown model.backend {config.model.backend!r}; expected 'dummy' or 'qwen_jlens'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RQ2 ablation analysis.")
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
        summary = run_rq2(config, model, families)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"backend {config.model.backend!r} cannot run yet: {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
