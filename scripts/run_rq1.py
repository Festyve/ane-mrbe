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
import dataclasses
import json
import sys

from jspace_binding.config import Config
from jspace_binding.experiments.rq1_probe import run_rq1
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.generate import generate_families


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RQ1 linear-probe analysis.")
    add_backend_args(parser)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "override experiment.seed AND the random-subspace capacity seed. "
            "Outputs redirect to <results>_S<seed>/ so they don't overwrite "
            "the seed-0 run. Use to check that the null is not an artifact of "
            "one random draw."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help=(
            "override model.layer_band with [L, L] and redirect outputs to "
            "<results>_L<layer>/. For the layer sweep: every result so far is "
            "at layer 48 alone, and 'the null holds at one layer' is a much "
            "weaker claim than 'across the workspace band' (raw 24-59 here). "
            "Reads only, no edits, so no refitting is needed."
        ),
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    capacity_seed: int | None = None
    if args.seed is not None:
        suffix = f"_S{args.seed}"
        config = dataclasses.replace(
            config,
            experiment=dataclasses.replace(config.experiment, seed=args.seed),
            paths=dataclasses.replace(
                config.paths,
                results=f"{config.paths.results}{suffix}",
                figures=f"{config.paths.figures}{suffix}",
            ),
        )
        capacity_seed = args.seed
        print(
            f"seed override: experiment.seed={args.seed} capacity_seed={args.seed} "
            f"results={config.paths.results} figures={config.paths.figures}",
            file=sys.stderr,
        )
    if args.layer is not None:
        # Outputs are redirected as well as the layer, so a sweep cannot
        # overwrite the layer-48 results already saved under runs/.
        suffix = f"_L{args.layer}"
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, layer_band=(args.layer, args.layer)),
            paths=dataclasses.replace(
                config.paths,
                results=f"{config.paths.results}{suffix}",
                figures=f"{config.paths.figures}{suffix}",
            ),
        )
        print(
            f"layer override: band={config.model.layer_band} "
            f"results={config.paths.results} figures={config.paths.figures}",
            file=sys.stderr,
        )
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    families = generate_families(config)
    print(f"generated {len(families)} families", file=sys.stderr)
    preflight_or_exit(model)  # probe reads need no fitted directions
    print(json.dumps(run_rq1(config, model, families, capacity_seed=capacity_seed), indent=2))


if __name__ == "__main__":
    main()
