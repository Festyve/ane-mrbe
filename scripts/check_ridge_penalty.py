#!/usr/bin/env python3
"""Is the cross-source comparison confounded by REGULARISATION? (RESULTS.md §8)

The open anomaly: `random_subspace` scores 0.719 at entity_token -- ABOVE
`residual` (0.545), the 5120-dimensional space it is a 16-dimensional
projection OF. A projection cannot contain information its source lacks, so
one of the two numbers is not measuring what it claims to.

The suspect is the fixed ridge penalty. `fit_ridge_probe` applies l2=1e-2 to
every source regardless of dimensionality, and 1e-2 is a very different
amount of regularisation for a 16-dim problem than for a 5120-dim one with
the same number of rows. The low-dimensional projection is then effectively
better regularised while the full residual overfits, and the ranking reports
that difference rather than any difference in content.

If so, every cross-source comparison in RQ1 inherits the confound -- so this
is not a footnote, it is a prerequisite for the table being readable at all.

This script collects RQ1's activations ONCE and re-runs the same
leave-one-pair-out probe over a grid of penalties, offline. One GPU pass, many
penalties.

READING THE OUTPUT. Three views, and the first is NOT the one to rank on:

  per-penalty table   every source at every penalty. Descriptive.
  best column         max over penalties -- OPTIMISTICALLY BIASED, because the
                      penalty is chosen on the same folds the accuracy is read
                      from. Never rank sources on this.
  nested CV           penalty chosen on inner folds, scored on the held-out
                      pair. The honest number, and the one to report.
  matched penalty     all sources at the penalty that suits the FULL-RANK
                      sources. This is the fair setting for the anomaly.

THE LOAD-BEARING CHECK is jspace against the CAPACITY CONTROL, not against
chance. The project never claimed jspace sits at exactly 0.5; it claimed the
workspace is not a privileged place for role, i.e. that jspace carries less
role information than an arbitrary subspace of the same rank. So the question
is whether jspace clears random_subspace at any penalty or under nested CV.
Checking jspace against an absolute chance floor instead would report an
overturned headline the moment heavy regularisation lifts every source --
which is a fact about probe capacity, not about where role lives.

THE ANOMALY (random_subspace above residual, the space it is a projection of)
must be judged at a MATCHED penalty. Comparing each source at its own best
lets the 16-dim source pick the setting where the 5120-dim ones are still
overfitting, which manufactures the impossibility rather than testing it.

    python scripts/check_ridge_penalty.py --config configs/default.yaml --limit 200
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jspace_binding.analysis.probes import (
    PROBE_SOURCES,
    ProbeExample,
    fit_ridge_probe,
    leave_one_pair_out,
)
from jspace_binding.config import Config
from jspace_binding.experiments.progress import track
from jspace_binding.model.factory import build_model, preflight_or_exit
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.types import InjectionSite, Position, Role

# Four orders of magnitude around the fixed 1e-2 the project has used
# everywhere. Wide on purpose: the claim being tested is that the optimum sits
# at a very different place for a 16-dim source than for a 5120-dim one, and a
# narrow grid could not show that even if it were true.
_PENALTIES: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
_FIXED = 1e-2  # what every other experiment used
# jspace must clear the capacity control by this much before "role is in
# J-space after all" is on the table. A margin, not a tie-break: two sources
# within noise of each other support no localisation claim either way.
_MARGIN = 0.02


def _nested_cv(rows: list[ProbeExample], penalties: tuple[float, ...], seed: int) -> float:
    """Leave-one-pair-out accuracy with the penalty chosen on INNER folds only.

    Reporting max-over-penalties from the outer folds — what the sweep table
    does — selects the hyperparameter on the same data it reports accuracy on,
    so every "best" column is optimistically biased and the bias is larger for
    the sources with more room to move. That is not a fair basis for ranking
    sources against each other, which is the only thing this script exists to
    do.

    Here each outer fold holds out one concept pair; the penalty is picked by
    a further leave-one-pair-out INSIDE the remaining pairs, then applied once
    to the held-out pair. The reported number is never chosen using the data
    it scores.
    """
    features = np.asarray([r.features for r in rows], dtype=float)
    labels = np.asarray([1.0 if r.is_agent else -1.0 for r in rows], dtype=float)
    groups = np.asarray([r.pair_id for r in rows])
    pair_ids = sorted(set(groups))
    if len(pair_ids) < 3:
        # Needs one pair for the outer fold and >= 2 inside it to select on.
        return float("nan")

    outer_accuracies = []
    for held_out in pair_ids:
        outer_test = groups == held_out
        outer_train = ~outer_test
        inner_ids = [p for p in pair_ids if p != held_out]

        best_l2, best_inner = penalties[0], -1.0
        for l2 in penalties:
            inner_scores = []
            for inner_held in inner_ids:
                inner_test = outer_train & (groups == inner_held)
                inner_train = outer_train & (groups != inner_held)
                if len(np.unique(labels[inner_train])) < 2 or not inner_test.any():
                    continue
                w = fit_ridge_probe(features[inner_train], labels[inner_train], l2=l2)
                augmented = np.hstack([features[inner_test], np.ones((inner_test.sum(), 1))])
                inner_scores.append(
                    float(np.mean(np.sign(augmented @ w) == labels[inner_test]))
                )
            mean_inner = float(np.mean(inner_scores)) if inner_scores else -1.0
            if mean_inner > best_inner:
                best_l2, best_inner = l2, mean_inner

        w = fit_ridge_probe(features[outer_train], labels[outer_train], l2=best_l2)
        augmented = np.hstack([features[outer_test], np.ones((outer_test.sum(), 1))])
        outer_accuracies.append(
            float(np.mean(np.sign(augmented @ w) == labels[outer_test]))
        )
    return float(np.mean(outer_accuracies)) if outer_accuracies else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--site", default=InjectionSite.ENTITY_TOKEN.value,
        choices=[s.value for s in InjectionSite],
        help="default entity_token: where the anomaly was observed and where "
             "role is actually decodable, so a jspace gain there cannot be "
             "dismissed as 'no signal anywhere'",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config)
    preflight_or_exit(model)
    site = InjectionSite(args.site)

    # Round-robin across pairs so --limit never silently covers one pair
    # (the same bug check_within_pair.py hit).
    grouped: dict[str, list] = defaultdict(list)
    for family in generate_families(config):
        grouped[family.concept_pair.pair_id].append(family)
    families = [f for group in zip(*grouped.values(), strict=False) for f in group][: args.limit]
    print(f"site={site.value}  families={len(families)}  "
          f"pairs={len({f.concept_pair.pair_id for f in families})}")

    rows: dict[str, list[ProbeExample]] = {s: [] for s in PROBE_SOURCES}
    for family in track(families, f"penalty {site.value}", total=len(families)):
        entity = family.concept_pair.entity
        for role in Role:
            for position in Position:
                activations = model.probe_activation(
                    family.cell(role, position).sentence, entity, site
                )
                for source in PROBE_SOURCES:
                    rows[source].append(
                        ProbeExample(
                            pair_id=family.concept_pair.pair_id,
                            is_agent=role is Role.AGENT,
                            features=tuple(float(x) for x in activations[source]),
                        )
                    )

    # One GPU pass above; everything below is closed-form numpy on cached rows.
    seed = config.experiment.seed
    table = {
        source: {
            l2: leave_one_pair_out(rows[source], l2=l2, control_seed=seed).accuracy
            for l2 in _PENALTIES
        }
        for source in PROBE_SOURCES
    }

    print(f"\nleave-one-pair-out accuracy by ridge penalty, chance = 0.50   "
          f"(site={site.value})\n")
    print(f"  {'source':18s}" + "".join(f"{l2:>10.0e}" for l2 in _PENALTIES) + f"{'best':>10s}")
    print("  " + "-" * (18 + 10 * len(_PENALTIES) + 10))
    for source, scores in table.items():
        best_l2 = max(scores, key=lambda k: scores[k])
        marker = "" if best_l2 == _FIXED else f"  <- best at {best_l2:.0e}"
        print(
            f"  {source:18s}"
            + "".join(f"{scores[l2]:10.3f}" for l2 in _PENALTIES)
            + f"{scores[best_l2]:10.3f}{marker}"
        )
    print(f"\n  (every other experiment used l2={_FIXED:.0e})")

    best = {source: max(scores.values()) for source, scores in table.items()}

    # Honest numbers: penalty chosen on inner folds, scored on the outer one.
    # The `best` column above is max-over-penalties on the SAME folds it
    # reports, so it cannot be used to rank sources (see _nested_cv).
    nested = {source: _nested_cv(rows[source], _PENALTIES, seed) for source in PROBE_SOURCES}
    print("\n  nested CV (penalty selected on inner folds — the honest number):")
    for source in PROBE_SOURCES:
        print(f"    {source:18s}{nested[source]:10.3f}")

    # The matched-penalty column. Comparing each source at its OWN best lets a
    # low-dimensional source pick the penalty where the high-dimensional ones
    # are still overfitting, which MANUFACTURES the projection-beats-source
    # anomaly rather than testing it. Everything high-dimensional peaks at the
    # strong end of the grid, so that is where the sources are comparable.
    matched_l2 = max(
        _PENALTIES,
        key=lambda l2: float(np.mean([table[s][l2] for s in ("residual", "orthogonal")])),
    )
    matched = {source: table[source][matched_l2] for source in PROBE_SOURCES}
    print(f"\n  at matched penalty l2={matched_l2:.0e} (best for the full-rank sources):")
    for source in PROBE_SOURCES:
        print(f"    {source:18s}{matched[source]:10.3f}")

    print("\n" + "=" * 68)

    # LOAD-BEARING CHECK. Not "is jspace above chance" -- the project's claim
    # was never that jspace sits at exactly 0.5. It is that jspace carries LESS
    # role information than an arbitrary subspace of the same rank, i.e. that
    # the workspace is not a privileged place for role. So the comparison is
    # against the capacity control, at every penalty, and under nested CV.
    beats_control_anywhere = [
        l2 for l2 in _PENALTIES if table["jspace"][l2] > table["random_subspace"][l2] + _MARGIN
    ]
    nested_beats_control = nested["jspace"] > nested["random_subspace"] + _MARGIN

    if beats_control_anywhere or nested_beats_control:
        headline = "jspace_beats_control_under_tuning"
        print("THE J-SPACE NULL DOES NOT SURVIVE PENALTY TUNING.")
        if beats_control_anywhere:
            print("  jspace exceeds the rank-matched control at l2 in "
                  f"{[f'{l2:.0e}' for l2 in beats_control_anywhere]}.")
        if nested_beats_control:
            print(f"  Under nested CV jspace {nested['jspace']:.3f} > control "
                  f"{nested['random_subspace']:.3f}.")
        print("  The null was an artifact of one regularisation choice. Every")
        print("  J-space claim needs rerunning at a tuned penalty before writing.")
    else:
        headline = "jspace_null_survives_penalty_sweep"
        rank = sorted(PROBE_SOURCES, key=lambda s: nested[s], reverse=True)
        print("THE J-SPACE NULL SURVIVES THE PENALTY SWEEP.")
        print(f"  jspace never clears the rank-matched control at any penalty in the")
        print(f"  grid, nor under nested CV ({nested['jspace']:.3f} vs "
              f"{nested['random_subspace']:.3f}).")
        print(f"  Nested-CV ranking: {' > '.join(rank)}.")
        print()
        print("  BUT RESTATE THE CLAIM. jspace moves from "
              f"{table['jspace'][_FIXED]:.3f} at l2={_FIXED:.0e} to")
        print(f"  {best['jspace']:.3f} at its best, so 'J-space is at chance' is too")
        print("  strong. The defensible claim is that J-space carries LESS role")
        print("  information than an arbitrary subspace of the same rank.")

    # The anomaly is about the cross-source RANKING only, and must be judged at
    # a matched penalty for the same reason the headline is.
    print()
    if matched["residual"] >= matched["random_subspace"] - _MARGIN:
        anomaly = "explained_by_regularisation"
        print("  ANOMALY EXPLAINED. At matched penalty "
              f"l2={matched_l2:.0e} residual ({matched['residual']:.3f})")
        print(f"  is level with or above random_subspace "
              f"({matched['random_subspace']:.3f}).")
        print("  'A projection beats its source' appears only at weak penalties,")
        print("  where the 5120-dim sources overfit and the 16-dim one does not.")
        print(f"  The fixed l2={_FIXED:.0e} was in that regime. Report matched-penalty or")
        print("  nested-CV numbers; the fixed-penalty ranking was confounded.")
    else:
        anomaly = "unexplained"
        print("  ANOMALY UNEXPLAINED. random_subspace leads residual even at matched")
        print(f"  penalty l2={matched_l2:.0e} ({matched['random_subspace']:.3f} vs "
              f"{matched['residual']:.3f}).")
        print("  Regularisation is ruled out; the cause is something else. Report it")
        print("  as an open anomaly rather than attributing it to the penalty.")
    print("=" * 68)

    out = Path(config.paths.results) / "ridge_penalty_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "site": site.value,
                "n_families": len(families),
                "penalties": list(_PENALTIES),
                "fixed_penalty": _FIXED,
                # str keys: JSON object keys must be strings, and 1e-2 would
                # otherwise round-trip as "0.01" inconsistently across writers.
                "accuracy": {s: {f"{l2:.0e}": a for l2, a in d.items()} for s, d in table.items()},
                "best_accuracy": best,
                "nested_cv_accuracy": nested,
                "matched_penalty": matched_l2,
                "matched_penalty_accuracy": matched,
                "headline": headline,
                "anomaly": anomaly,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
