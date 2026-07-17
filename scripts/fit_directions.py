#!/usr/bin/env python3
"""Generate the fitting corpus, cache activations, fit role directions.

Usage: fit_directions.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]

Runs the one-time direction-fitting pass (proposal, Methods / Role directions):
for every entity that any condition pushes (the concept-pair entities plus the
non-participant candidates), at every configured injection site, fit
r_entity = mean(agent) - mean(patient) over J-space activations, plus the
shuffled-label and leave-one-out generic control variants, and run the
bootstrap stability pilot check. Directions land in config.paths.directions;
the fitting corpus is archived to config.paths.fitting_corpus.

--dry-run forces the GPU-free DummyModel, whose synthetic activations carry a
planted role direction in `binding` mode (stability should come out high) and
pure noise in `bag` mode (the stability warning should fire — that is the
pilot check working, not a bug).

Progress goes to stderr; stdout carries only the JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from jspace_binding.config import Config
from jspace_binding.directions.fit import fit_all, save_directions
from jspace_binding.model.base import FittingActivationSource
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.fitting_corpus import generate_fitting_corpus, save_fitting_corpus
from jspace_binding.types import Role


def _build_model(config: Config, dry_run: bool, dummy_mode: str | None) -> FittingActivationSource:
    backend = "dummy" if dry_run else config.model.backend
    if backend == "dummy":
        return DummyModel(mode=dummy_mode or config.model.dummy_mode, seed=config.experiment.seed)
    if backend == "qwen_jlens":
        from jspace_binding.model.qwen_jlens import QwenJLensModel

        return QwenJLensModel(config.model, directions_dir=config.paths.directions)
    sys.exit(f"unknown model.backend {config.model.backend!r}; expected 'dummy' or 'qwen_jlens'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit role directions from the fitting corpus.")
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
    save_fitting_corpus(corpus, config.paths.fitting_corpus)
    print(
        f"fitting corpus: {len(corpus)} sentences ({len(entities)} entities x 2 roles "
        f"x {config.directions.exemplars_per_role}) -> {config.paths.fitting_corpus}",
        file=sys.stderr,
    )

    summary: dict[str, object] = {"entities": list(entities), "sites": {}}
    for site in config.experiment.injection_sites:
        activations = {}
        for entity in entities:
            rows = {role: [] for role in Role}
            for ex in corpus:
                if ex.entity == entity:
                    rows[ex.role].append(model.fitting_activation(ex.sentence, entity, site))
            activations[entity] = (
                np.asarray(rows[Role.AGENT], dtype=float),
                np.asarray(rows[Role.PATIENT], dtype=float),
            )
        directions = fit_all(
            activations,
            site,
            n_bootstrap=config.directions.n_bootstrap,
            seed=config.directions.seed,
        )
        path = save_directions(directions, config.paths.directions)
        site_summary = {
            "path": str(path),
            "stability": {
                entity: float(directions.stability[i])
                for i, entity in enumerate(directions.entities)
            },
            "raw_norms": {
                entity: float(directions.raw_norms[i])
                for i, entity in enumerate(directions.entities)
            },
        }
        summary["sites"][site.value] = site_summary
        for entity, stability in site_summary["stability"].items():
            if stability < config.directions.stability_threshold:
                print(
                    f"WARNING [{site.value}] r_{entity} bootstrap stability "
                    f"{stability:.3f} < {config.directions.stability_threshold} — "
                    "fitting corpus too small/noisy, or no role signal at this site "
                    "(the proposal's pilot check; grow exemplars_per_role before trusting "
                    "any binding score built on this direction)",
                    file=sys.stderr,
                )
        print(f"[{site.value}] fitted {len(entities)} directions -> {path}", file=sys.stderr)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
