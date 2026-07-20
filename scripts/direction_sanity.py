#!/usr/bin/env python3
"""Sanity-check fitted role-directions on held-out sentences.

Usage: direction_sanity.py --config configs/default.yaml [--dry-run]
                           [--dummy-mode binding|bag] [--eval-corpus PATH]
                           [--entity doctor,nurse]

Loads the directions_{site}.npz that scripts/fit_directions.py wrote and asks
the two pilot questions (analysis.direction_sanity):

1. Held-out separation — project held-out agent/patient activations onto each
   entity's fitted unit direction; report AUC, midpoint-threshold accuracy,
   and Cohen's d, plus a strip-scatter figure per site (skipped gracefully if
   matplotlib is unavailable).
2. Direction comparison — pairwise cosine between the per-entity fitted
   directions at each site, with a plain-language verdict (same vector /
   related but distinct / distinct).

Held-out sentences come from --eval-corpus (a FittingExample JSONL such as
data/handwritten/eval_doctor.jsonl) or, by default, are generated from the
frame templates at twice the configured exemplars_per_role and stripped of
every sentence in the archived fitting corpus (config.paths.fitting_corpus) —
disjoint by construction and re-checked verbatim, so the check never grades
the direction on sentences it was fit on.

--dry-run uses the GPU-free DummyModel: `binding` mode plants a role axis
(separation should be near-perfect — the harness validating itself), `bag`
mode is pure noise (AUC should sit at chance; that is the check working).

Progress goes to stderr; stdout carries only the JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from jspace_binding.analysis.direction_sanity import (
    compare_directions,
    project,
    scatter_projections,
    separation_report,
)
from jspace_binding.config import Config
from jspace_binding.directions.export_pt import load_fitted_by_site
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.fitting_corpus import (
    generate_fitting_corpus,
    load_fitting_corpus,
)
from jspace_binding.types import Role


def _held_out_corpus(config: Config, eval_corpus: Path | None, entities: tuple[str, ...]):
    """Held-out FittingExamples, guaranteed disjoint from the fitting set."""
    fitting_sentences: set[str] = set()
    if config.paths.fitting_corpus.exists():
        fitting_sentences = {ex.sentence for ex in load_fitting_corpus(config.paths.fitting_corpus)}
    elif eval_corpus is None:
        sys.exit(
            f"no archived fitting corpus at {config.paths.fitting_corpus} — run "
            "scripts/fit_directions.py first, or pass --eval-corpus explicitly"
        )

    if eval_corpus is not None:
        examples = load_fitting_corpus(eval_corpus)
        source = str(eval_corpus)
    else:
        # The generator is deterministic and cycles frames -> verbs -> distractors,
        # so doubling exemplars_per_role yields a superset; dropping the archived
        # fitting sentences leaves genuinely unseen verb/distractor combinations.
        examples = generate_fitting_corpus(entities, 2 * config.directions.exemplars_per_role)
        source = "generated (2x exemplars_per_role minus the fitting set)"

    held_out = [ex for ex in examples if ex.sentence not in fitting_sentences]
    dropped = len(examples) - len(held_out)
    # Re-check verbatim: grading on fitting sentences would inflate separation.
    overlap = {ex.sentence for ex in held_out} & fitting_sentences
    assert not overlap, f"held-out set still overlaps fitting corpus: {sorted(overlap)[:3]}"
    print(
        f"held-out corpus: {len(held_out)} sentences from {source}"
        + (f" ({dropped} fitting-set sentences dropped)" if dropped else ""),
        file=sys.stderr,
    )
    return held_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check fitted role-directions.")
    add_backend_args(parser)
    parser.add_argument(
        "--eval-corpus",
        type=Path,
        default=None,
        help="held-out FittingExample JSONL (default: generate frames beyond the fitting set)",
    )
    parser.add_argument(
        "--entity",
        type=lambda s: tuple(e.strip() for e in s.split(",") if e.strip()),
        default=None,
        help="comma-separated entities to check (default: every fitted entity)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    directions_by_site = load_fitted_by_site(config.paths.directions)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    preflight_or_exit(model, tuple(directions_by_site))

    fitted_entities = sorted({e for d in directions_by_site.values() for e in d.entities})
    entities = args.entity or tuple(fitted_entities)
    missing = [e for e in entities if e not in fitted_entities]
    if missing:
        sys.exit(f"no fitted direction for: {', '.join(missing)}; fitted: {fitted_entities}")

    corpus = _held_out_corpus(config, args.eval_corpus, entities)
    summary: dict[str, object] = {"held_out_sentences": len(corpus), "sites": {}}

    for site, fitted in directions_by_site.items():
        site_summary: dict[str, object] = {"separation": {}, "comparison": None}
        scatter_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for entity in entities:
            if entity not in fitted.entities:
                continue
            rows = {role: [] for role in Role}
            for ex in corpus:
                if ex.entity == entity:
                    rows[ex.role].append(model.fitting_activation(ex.sentence, entity, site))
            if not rows[Role.AGENT] or not rows[Role.PATIENT]:
                print(f"[{site.value}] {entity}: no held-out sentences; skipped", file=sys.stderr)
                continue
            direction = fitted.direction(entity, "fitted")
            agent_proj = project(np.asarray(rows[Role.AGENT], float), direction)
            patient_proj = project(np.asarray(rows[Role.PATIENT], float), direction)
            report = separation_report(agent_proj, patient_proj, entity, site.value)
            scatter_data[f"{entity}@{site.value}"] = (agent_proj, patient_proj)
            site_summary["separation"][entity] = {
                "n_agent": report.n_agent,
                "n_patient": report.n_patient,
                "auc": report.auc,
                "accuracy": report.accuracy,
                "cohens_d": report.cohens_d,
                "separates": report.separates,
            }
            print(
                f"[{site.value}] {entity}: AUC {report.auc:.3f}, "
                f"acc {report.accuracy:.3f}, d {report.cohens_d:.2f} "
                f"({report.n_agent} agent / {report.n_patient} patient held-out)",
                file=sys.stderr,
            )

        checked = [e for e in entities if e in fitted.entities]
        if len(checked) >= 2:
            site_summary["comparison"] = compare_directions(
                {e: fitted.direction(e, "fitted") for e in checked}
            )
            print(
                f"[{site.value}] direction comparison: "
                f"mean cosine {site_summary['comparison']['mean_off_diagonal_cosine']:.3f} "
                f"-> {site_summary['comparison']['verdict']}",
                file=sys.stderr,
            )

        if scatter_data:
            fig_path = scatter_projections(
                scatter_data, Path(config.paths.figures) / f"direction_sanity_{site.value}.png"
            )
            site_summary["figure"] = None if fig_path is None else str(fig_path)
            if fig_path is None:
                print(f"[{site.value}] matplotlib unavailable; scatter skipped", file=sys.stderr)

        summary["sites"][site.value] = site_summary

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
