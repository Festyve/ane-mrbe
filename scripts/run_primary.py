#!/usr/bin/env python3
"""Run the primary experiment end-to-end and print the JSON summary to stdout.

Usage: run_primary.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]

--dry-run forces the GPU-free DummyModel regardless of config.model.backend, so
the whole pipeline can be validated on any machine. Exits with status 2 when
the qwen_jlens backend is not runnable yet (open config decisions, missing
lens artifact, or directions not fitted — run scripts/fit_directions.py and
scripts/calibrate.py first); that check runs BEFORE the sweep, so a failure
mid-sweep is a genuine error, not a configuration problem. Progress messages
go to stderr; stdout carries only the summary.
"""

from __future__ import annotations

import argparse
import json
import sys

from jspace_binding.config import Config
from jspace_binding.experiments.primary import analyze, run_primary, validate_config
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.generate import generate_families


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the primary binding experiment.")
    add_backend_args(parser)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    validate_config(config)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    # Always regenerate: generation is deterministic and effectively free, and
    # loading a stale stimuli.jsonl written under a different config would
    # silently win. scripts/generate_stimuli.py remains the archival path.
    families = generate_families(config)
    print(f"generated {len(families)} families", file=sys.stderr)
    preflight_or_exit(model, config.experiment.injection_sites)
    trials = run_primary(config, model, families)
    print(json.dumps(analyze(config, trials), indent=2))


if __name__ == "__main__":
    main()
