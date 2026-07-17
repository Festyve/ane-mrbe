#!/usr/bin/env python3
"""Run RQ1 (role-decodability linear probes) and print the JSON summary.

Usage: run_rq1.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]

Caches no-edit activations for every primary-stimulus cell, probes each
(injection site x activation source) with the leave-one-pair-out ridge probe,
and writes rq1_probe.json + the selectivity figure. --dry-run uses the
DummyModel's planted ground truth: binding mode puts the role signal in the
J-space component, bag mode in the orthogonal remainder. Exits 2 when the
real backend is not runnable yet (checked before the sweep).
"""

from __future__ import annotations

import argparse
import json
import sys

from jspace_binding.config import Config
from jspace_binding.experiments.rq1_probe import run_rq1
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.generate import generate_families


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RQ1 linear-probe analysis.")
    add_backend_args(parser)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    families = generate_families(config)
    print(f"generated {len(families)} families", file=sys.stderr)
    preflight_or_exit(model)  # probe reads need no fitted directions
    print(json.dumps(run_rq1(config, model, families), indent=2))


if __name__ == "__main__":
    main()
