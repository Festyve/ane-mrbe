#!/usr/bin/env python3
"""Calibrate the two steering strengths and write data/calibration.json.

Usage: calibrate.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]

- push_coefficient: smallest grid value whose toward-agent push detectably
  moves patient-role FITTING-CORPUS sentences (never the primary stimuli).
- alpha (IDENTITY_SWAP): smallest grid value that moves the NEUTRAL-probe
  readout on a small stimulus sample.

Copy the printed values into configs/*.yaml (model.push_coefficient /
model.alpha) before running the primary experiment — the runner reads them
from config, not from calibration.json (the JSON is the audit record).

Exit codes: 2 = backend not runnable yet (missing config/lens/directions);
3 = intervention-strength failure — no grid value moved behavior enough,
which is the proposal's uninterpretable-null outcome, not a crash.

The DummyModel has no strength dial, so --dry-run returns the smallest grid
values; this script's dry-run exists to exercise the plumbing end-to-end.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from jspace_binding.config import Config
from jspace_binding.experiments.calibrate import (
    calibrate_identity_alpha,
    calibrate_push_coefficient,
)
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.fitting_corpus import generate_fitting_corpus
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.types import InjectionSite


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate push_coefficient and alpha.")
    add_backend_args(parser)
    parser.add_argument(
        "--site",
        choices=[s.value for s in InjectionSite],
        default=InjectionSite.FINAL_TOKEN.value,
        help=(
            "site whose fitted directions to calibrate. Was hard-coded to "
            "final_token; must match the site run_primary will sweep, and "
            "should be a site whose directions passed the stability check in "
            "fit_directions -- calibrating an unstable direction tunes the "
            "strength of a vector that is not reproducible"
        ),
    )
    parser.add_argument(
        "--push-grid",
        type=lambda s: tuple(float(v) for v in s.split(",")),
        default=None,
        help=(
            "comma-separated push_coefficient grid, e.g. 8,16,32,64,128. The "
            "coefficient is in ABSOLUTE residual-norm units (edits.py), so the "
            "default grid does not transfer across models with different "
            "activation scales -- Qwen3.6-27B already needed the default's "
            "largest value. Legitimate to extend: calibration reads the "
            "fitting corpus only, never the primary stimuli."
        ),
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    site = InjectionSite(args.site)
    print(f"calibrating at site={site.value}", file=sys.stderr)
    preflight_or_exit(model, (site,))

    corpus = generate_fitting_corpus(
        config.direction_entities(),
        config.directions.exemplars_per_role,
        counterparts=config.counterpart_entities(),
    )
    families = generate_families(config)

    try:
        push_kwargs = {"grid": args.push_grid} if args.push_grid else {}
        push = calibrate_push_coefficient(model, corpus, site=site, **push_kwargs)
        alpha = calibrate_identity_alpha(model, families, site=site)
    except ValueError as exc:
        # The documented failure mode: the intervention is too weak at every
        # grid value, so any null binding result would be uninterpretable.
        print(f"intervention-strength failure: {exc}", file=sys.stderr)
        sys.exit(3)

    record = {
        # Which site these were calibrated at. A coefficient tuned at one site
        # does not transfer to another -- the fitted directions differ in norm
        # (measured ~2.2-2.8 at final_token vs ~5.7-7.3 at entity_token on
        # Qwen3.6-27B), so the same coefficient is a different push.
        "site": site.value,
        "push_coefficient": asdict(push),
        "alpha": asdict(alpha),
        "note": (
            "copy push_coefficient.value / alpha.value into configs/*.yaml "
            "(model section), then run run_primary.py --site " + site.value
        ),
    }
    out = Path(config.paths.calibration)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
