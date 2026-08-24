#!/usr/bin/env python3
"""Run the primary experiment end-to-end and print the JSON summary to stdout.

Usage: run_primary.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]
                      [--site final_token|entity_token]

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
from jspace_binding.experiments.primary import (
    analyze,
    resolve_calibrated_strengths,
    run_primary,
    validate_config,
)
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.types import InjectionSite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the primary binding experiment.")
    add_backend_args(parser)
    parser.add_argument(
        "--site",
        choices=[s.value for s in InjectionSite],
        default=None,
        help=(
            "restrict the sweep to one injection site (default: every site in "
            "experiment.injection_sites). analyze() scores a single site anyway, "
            "so naming the one you will analyze halves GPU cost"
        ),
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    validate_config(config)
    site = InjectionSite(args.site) if args.site else None
    # Load the calibrated alpha / push_coefficient from data/calibration.json
    # when the config leaves them open, so the calibrate -> primary handoff no
    # longer depends on a manual YAML edit (and cannot silently degrade to a
    # pure swap). Bypassed under --dry-run. Exits 2 if a push experiment has no
    # usable coefficient.
    config = resolve_calibrated_strengths(config, site, dry_run=args.dry_run)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    # Always regenerate: generation is deterministic and effectively free, and
    # loading a stale stimuli.jsonl written under a different config would
    # silently win. scripts/generate_stimuli.py remains the archival path.
    families = generate_families(config)
    print(f"generated {len(families)} families", file=sys.stderr)
    # Only the swept sites need fitted directions.
    preflight_or_exit(model, (site,) if site else config.experiment.injection_sites)
    trials = run_primary(config, model, families, site=site)
    print(json.dumps(analyze(config, trials, site=site), indent=2))


if __name__ == "__main__":
    main()
