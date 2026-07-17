#!/usr/bin/env python3
"""Run RQ2 (J-space ablation vs matched controls) and print the JSON summary.

Usage: run_rq2.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]

Scores the binding task (ROLE probe) and the difficulty-matched recall task
(NEUTRAL probe) under NO_EDIT / ABLATE_JSPACE / ABLATE_RANDOM_SUBSPACE, and
writes rq2_ablation.json + the ablation-deltas figure. --dry-run uses the
DummyModel's planted ground truth: binding mode shows a binding-specific
J-space deficit exceeding the random-subspace bar; bag mode does not. Exits 2
when the real backend is not runnable yet (checked before the sweep).
"""

from __future__ import annotations

import argparse
import json
import sys

from jspace_binding.config import Config
from jspace_binding.experiments.rq2_ablation import run_rq2
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.generate import generate_families


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RQ2 ablation analysis.")
    add_backend_args(parser)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    families = generate_families(config)
    print(f"generated {len(families)} families", file=sys.stderr)
    preflight_or_exit(model)  # ablations need the lens but no fitted directions
    print(json.dumps(run_rq2(config, model, families), indent=2))


if __name__ == "__main__":
    main()
