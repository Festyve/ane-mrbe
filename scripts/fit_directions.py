#!/usr/bin/env python3
"""Generate the fitting corpus, cache activations, fit role directions.

Usage: fit_directions.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]
                         [--corpus PATH] [--allow-contaminated]

Runs the one-time direction-fitting pass (proposal, Methods / Role directions):
for every entity that any condition pushes (the concept-pair entities plus the
non-participant candidates), at every configured injection site, fit
r_entity = mean(agent) - mean(patient) over J-space activations, plus the
shuffled-label and leave-one-out generic control variants, and run the
bootstrap stability pilot check. Directions land in config.paths.directions;
the fitting corpus is archived to config.paths.fitting_corpus.

--corpus PATH fits from a hand-written JSONL instead of the generated frames.
The corpus is refused if any of its sentences occur verbatim in the primary
stimulus set at this config's scale (fitting and testing on the same
sentences contaminates the causal test); --allow-contaminated overrides for
throwaway integration tests and stamps "contaminated": true into the summary.

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
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.fitting_corpus import (
    generate_fitting_corpus,
    load_fitting_corpus,
    save_fitting_corpus,
)
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.stimuli.qc import find_primary_collisions
from jspace_binding.types import Role


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit role directions from the fitting corpus.")
    add_backend_args(parser)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="hand-written fitting corpus JSONL (default: generate from the frame templates)",
    )
    parser.add_argument(
        "--allow-contaminated",
        action="store_true",
        help="proceed despite primary-set collisions (integration tests only; stamped in output)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    preflight_or_exit(model)  # fitting itself needs no already-fitted directions

    needed = config.direction_entities()
    contaminated = False
    if args.corpus is not None:
        corpus = load_fitting_corpus(args.corpus)
        entities = tuple(sorted({ex.entity for ex in corpus}))
        missing = [e for e in needed if e not in entities]
        if missing:
            print(
                f"WARNING --corpus covers only {list(entities)}; the primary experiment "
                f"also needs directions for {missing} (fine for an integration test, "
                "not for the full sweep)",
                file=sys.stderr,
            )
        collisions = find_primary_collisions(corpus, generate_families(config))
        if collisions:
            contaminated = True
            print(
                f"{len(collisions)} corpus sentences occur verbatim in the primary "
                "stimulus set at this config's scale — fitting and testing on the same "
                "sentences contaminates the causal test (scripts/check_corpus.py lists "
                "them).",
                file=sys.stderr,
            )
            if not args.allow_contaminated:
                print("refusing to fit; fix the corpus or pass --allow-contaminated",
                      file=sys.stderr)
                sys.exit(1)
        print(f"fitting corpus: {len(corpus)} sentences (hand-written, {args.corpus})",
              file=sys.stderr)
    else:
        entities = needed
        corpus = generate_fitting_corpus(entities, config.directions.exemplars_per_role)
        save_fitting_corpus(corpus, config.paths.fitting_corpus)
        print(
            f"fitting corpus: {len(corpus)} sentences ({len(entities)} entities x 2 roles "
            f"x {config.directions.exemplars_per_role}) -> {config.paths.fitting_corpus}",
            file=sys.stderr,
        )

    summary: dict[str, object] = {
        "entities": list(entities),
        "contaminated": contaminated,
        "sites": {},
    }
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
