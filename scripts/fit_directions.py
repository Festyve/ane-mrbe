#!/usr/bin/env python3
"""Generate the fitting corpus, cache activations, fit role directions.

Usage: fit_directions.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]
                         [--corpus PATH] [--allow-contaminated] [--entities a,b]
                         [--directions-dir PATH] [--fitting-corpus-out PATH]

Runs the one-time direction-fitting pass (proposal, Methods / Role directions):
for every entity that any condition pushes (the concept-pair entities plus the
non-participant candidates), at every configured injection site, fit
r_entity = mean(agent) - mean(patient) over J-space activations, plus the
shuffled-label and leave-one-out generic control variants, and run the
bootstrap stability pilot check. Directions land in config.paths.directions;
a partial --entities fit is routed to a tagged subdirectory under it so it
cannot overwrite a canonical full fit;
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
import hashlib
import json
import subprocess
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


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _export_metadata(config: Config, corpus_path: Path, directions_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": config.model.model_id,
        "lens_repo": config.model.lens_repo,
        "layer_band": None
        if config.model.layer_band is None
        else list(config.model.layer_band),
        "injection_sites": [site.value for site in config.experiment.injection_sites],
        "direction_variant": config.directions.variant,
        "directions_dir": str(directions_dir),
        "fitting_corpus": str(corpus_path),
        "fitting_corpus_sha256": _sha256(corpus_path),
        "exemplars_per_role": config.directions.exemplars_per_role,
        "bootstrap_resamples": config.directions.n_bootstrap,
        "direction_seed": config.directions.seed,
        "push_coefficient": config.model.push_coefficient,
        "git_commit": _git_commit(),
    }


def _tagged_directions_dir(base: Path, tag: str) -> Path:
    """Route a partial fit to <base>/partial/<tag>, without re-nesting.

    Silently dangerous otherwise: pointing paths.directions at a previous
    partial run appends a SECOND partial/<tag>, so the fit writes to the
    deeper path while preflight and scripts/direction_sanity.py keep reading
    paths.directions — and quietly score a stale .npz from an earlier run.
    Observed for real: a re-fit reported new stability values while the
    cosine check still returned the pre-fix numbers.
    """
    if base.parts[-2:] == ("partial", tag):
        return base  # already the tagged directory; reuse it
    if "partial" in base.parts:
        sys.exit(
            f"paths.directions ({base}) is already inside a partial-fit directory, but "
            f"this run is tagged {tag!r}. Point it at the canonical directions directory "
            "or pass --directions-dir explicitly."
        )
    return base / "partial" / tag


def _tagged_corpus_path(base: Path, tag: str) -> Path:
    """<stem>_<tag><suffix>, without double-tagging an already-tagged file."""
    if base.stem.endswith(f"_{tag}"):
        return base
    return base.with_name(f"{base.stem}_{tag}{base.suffix}")


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
    parser.add_argument(
        "--export-pt",
        action="store_true",
        help="also write {entity}_role_direction.pt (Group A handoff; needs the torch extra)",
    )
    parser.add_argument(
        "--entities",
        type=lambda s: tuple(e.strip() for e in s.split(",") if e.strip()),
        default=None,
        help="fit these entities instead of config.direction_entities() "
        "(e.g. doctor,nurse for a pair handoff; ignored with --corpus)",
    )
    parser.add_argument(
        "--directions-dir",
        type=Path,
        default=None,
        help="output directory for directions (partial fits default to a safe subdirectory)",
    )
    parser.add_argument(
        "--fitting-corpus-out",
        type=Path,
        default=None,
        help="where to archive the fitting corpus (partial fits default to a safe filename)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    preflight_or_exit(model)  # fitting itself needs no already-fitted directions

    needed = config.direction_entities()
    partial_entities = bool(
        args.corpus is None and args.entities and set(args.entities) != set(needed)
    )
    directions_dir = args.directions_dir or config.paths.directions
    fitting_corpus_path = args.fitting_corpus_out or args.corpus or config.paths.fitting_corpus
    if partial_entities:
        tag = "-".join(sorted(args.entities))
        directions_dir = args.directions_dir or _tagged_directions_dir(
            config.paths.directions, tag
        )
        fitting_corpus_path = args.fitting_corpus_out or _tagged_corpus_path(
            config.paths.fitting_corpus, tag
        )
        print(
            f"partial fit: writing to {directions_dir} and {fitting_corpus_path}; "
            "the canonical full-fit outputs will not be overwritten",
            file=sys.stderr,
        )
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
        entities = args.entities or needed
        if args.entities:
            missing = [e for e in needed if e not in args.entities]
            if missing:
                print(
                    f"WARNING --entities covers only {list(args.entities)}; the primary "
                    f"experiment also needs directions for {missing}",
                    file=sys.stderr,
                )
        corpus = generate_fitting_corpus(
            entities,
            config.directions.exemplars_per_role,
            counterparts=config.counterpart_entities(),
        )
        save_fitting_corpus(corpus, fitting_corpus_path)
        print(
            f"fitting corpus: {len(corpus)} sentences ({len(entities)} entities x 2 roles "
            f"x {config.directions.exemplars_per_role}) -> {fitting_corpus_path}",
            file=sys.stderr,
        )
    if args.corpus is not None and args.fitting_corpus_out is not None:
        save_fitting_corpus(corpus, fitting_corpus_path)

    summary: dict[str, object] = {
        "entities": list(entities),
        "contaminated": contaminated,
        "directions_dir": str(directions_dir),
        "fitting_corpus": str(fitting_corpus_path),
        "sites": {},
    }
    fitted_by_site = {}
    use_gradient = config.directions.estimator == "lre_gradient"
    for site in config.experiment.injection_sites:
        activations = {}
        for entity in entities:
            rows = {role: [] for role in Role}
            for ex in corpus:
                if ex.entity == entity:
                    if use_gradient:
                        rows[ex.role].append(
                            model.fitting_gradient(
                                ex.sentence, ex.role_probe, entity, ex.other, site
                            )
                        )
                    else:
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
            estimator=config.directions.estimator,
        )
        fitted_by_site[site] = directions
        path = save_directions(directions, directions_dir)
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

    if args.export_pt:
        from jspace_binding.directions.export_pt import export_pt

        try:
            written = export_pt(
                fitted_by_site,
                directions_dir,
                metadata=_export_metadata(config, fitting_corpus_path, directions_dir),
            )
        except ModuleNotFoundError as exc:
            # The .npz fit already succeeded; a missing torch shouldn't fail the run.
            print(f"WARNING --export-pt skipped: {exc}", file=sys.stderr)
            summary["pt_export"] = {"skipped": str(exc)}
        else:
            print(f"exported {len(written)} .pt files -> {directions_dir}",
                  file=sys.stderr)
            summary["pt_export"] = {"written": [str(p) for p in written]}

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
