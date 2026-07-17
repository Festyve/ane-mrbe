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
from jspace_binding.model.base import WorkspaceModel
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.fitting_corpus import generate_fitting_corpus
from jspace_binding.stimuli.generate import generate_families


def _build_model(config: Config, dry_run: bool, dummy_mode: str | None) -> WorkspaceModel:
    backend = "dummy" if dry_run else config.model.backend
    if backend == "dummy":
        return DummyModel(mode=dummy_mode or config.model.dummy_mode, seed=config.experiment.seed)
    if backend == "qwen_jlens":
        from jspace_binding.model.qwen_jlens import QwenJLensModel

        return QwenJLensModel(
            config.model,
            directions_dir=config.paths.directions,
            direction_variant=config.directions.variant,
        )
    sys.exit(f"unknown model.backend {config.model.backend!r}; expected 'dummy' or 'qwen_jlens'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate push_coefficient and alpha.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--dry-run", action="store_true", help="force the GPU-free DummyModel backend"
    )
    parser.add_argument("--dummy-mode", choices=("binding", "bag"), default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = _build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)

    entities = tuple(
        dict.fromkeys(
            [pair.entity for pair in config.stimuli.concept_pairs]
            + list(config.stimuli.non_participant_entities)
        )
    )
    corpus = generate_fitting_corpus(entities, config.directions.exemplars_per_role)
    families = generate_families(config)

    try:
        push = calibrate_push_coefficient(model, corpus)
        alpha = calibrate_identity_alpha(model, families)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"calibration cannot run yet: {exc}", file=sys.stderr)
        sys.exit(2)

    record = {
        "push_coefficient": asdict(push),
        "alpha": asdict(alpha),
        "note": "copy push_coefficient.value / alpha.value into configs/*.yaml (model section)",
    }
    out = Path(config.paths.calibration)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
