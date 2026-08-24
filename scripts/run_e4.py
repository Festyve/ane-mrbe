#!/usr/bin/env python3
"""Run E4 (on-demand recruitment) and print the JSON summary.

Usage: run_e4.py --config configs/default.yaml [--dry-run]
                 [--dummy-mode binding|bag|recruitment]

Presents identical stimulus tokens under a role question and a bag question and
measures how decodable role is from J-space in each. The delta separates three
hypotheses the other experiments conflate:

    always-on binding : both conditions high, delta ~ 0
    recruited         : role high, bag at chance, delta large
    absent            : both at chance

--dummy-mode recruitment matters here: binding and recruitment mode are
identical to every other experiment and differ only at this read, which is why
E4 exists. Reads at the final token of sentence+question, since any in-sentence
position is question-independent under a causal mask. Exits 2 when the real
backend is not runnable yet.

Cost: 4,800 forward passes at the default 600 families, the same order as RQ1.
"""

from __future__ import annotations

import argparse
import json
import sys

from jspace_binding.config import Config
from jspace_binding.experiments.recruitment import run_e4
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.generate import generate_families


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the E4 recruitment analysis.")
    add_backend_args(parser)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    families = generate_families(config)
    print(f"generated {len(families)} families", file=sys.stderr)
    preflight_or_exit(model)  # probe reads need no fitted directions
    print(json.dumps(run_e4(config, model, families), indent=2))


if __name__ == "__main__":
    main()
