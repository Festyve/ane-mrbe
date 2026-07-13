#!/usr/bin/env python3
"""Run the primary experiment end-to-end and print the JSON summary to stdout.

Usage: run_primary.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]

--dry-run forces the GPU-free DummyModel regardless of config.model.backend, so
the whole pipeline can be validated on any machine. Exits with status 2 when
the requested backend is the not-yet-implemented qwen_jlens stub. Progress
messages go to stderr; stdout carries only the summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jspace_binding.config import Config
from jspace_binding.experiments.primary import analyze, run_primary
from jspace_binding.model.base import WorkspaceModel
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.types import ItemFamily


def _build_model(config: Config, dry_run: bool, dummy_mode: str | None) -> WorkspaceModel:
    backend = "dummy" if dry_run else config.model.backend
    if backend == "dummy":
        return DummyModel(mode=dummy_mode or config.model.dummy_mode, seed=config.experiment.seed)
    if backend == "qwen_jlens":
        # Imported lazily: the only backend that will ever need the heavy `model` extras.
        from jspace_binding.model.qwen_jlens import QwenJLensModel

        return QwenJLensModel(config.model)
    sys.exit(f"unknown model.backend {config.model.backend!r}; expected 'dummy' or 'qwen_jlens'")


def _load_or_generate(config: Config) -> list[ItemFamily]:
    # Always regenerate: generation is deterministic and effectively free, and
    # loading a stale stimuli.jsonl written under a different config would
    # silently win. scripts/generate_stimuli.py remains the archival path.
    families = generate_families(config)
    print(f"generated {len(families)} families", file=sys.stderr)
    return families


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the primary binding experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--dry-run", action="store_true", help="force the GPU-free DummyModel backend"
    )
    parser.add_argument(
        "--dummy-mode",
        choices=("binding", "bag"),
        default=None,
        help="DummyModel ground truth (default: config model.dummy_mode)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = _build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    families = _load_or_generate(config)
    try:
        trials = run_primary(config, model, families)
    except NotImplementedError as exc:
        # qwen_jlens stub path: construction succeeds, the first forward raises.
        print(f"backend {config.model.backend!r} is not implemented yet: {exc}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(analyze(config, trials), indent=2))


if __name__ == "__main__":
    main()
