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

READING THE OUTPUT. Compare each source at its OWN best penalty, which is the
comparison the fixed-penalty table failed to make:

  residual overtakes random_subspace  -> the anomaly WAS a regularisation
                                         artifact. The J-space null is
                                         unaffected (see below) but the
                                         cross-source ranking needs restating.
  random_subspace still on top        -> not regularisation. The anomaly is
                                         real and needs a different account;
                                         report it as open.

THE LOAD-BEARING CHECK is jspace. Every conclusion in this project rests on
J-space sitting at chance, so the question that matters is not which source
wins but whether jspace clears chance under ANY penalty. If it does, the
headline was a regularisation artifact and the paper changes. If it does not
-- if jspace is at chance across four orders of magnitude of penalty -- then
no regularisation story explains the null away, and the anomaly is confined
to the cross-source ranking.

    python scripts/check_ridge_penalty.py --config configs/default.yaml --limit 200
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jspace_binding.analysis.probes import PROBE_SOURCES, ProbeExample, leave_one_pair_out
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
# jspace must clear chance by this much at SOME penalty before the null could
# be called a regularisation artifact. Same floor as the nonlinear probe, for
# the same reason: a relative rule with no absolute bar always finds a winner.
_MIN_ABOVE_CHANCE = 0.08


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
    jspace_best = best["jspace"]
    jspace_best_l2 = max(table["jspace"], key=lambda k: table["jspace"][k])

    print("\n" + "=" * 68)

    # The load-bearing check FIRST: the headline null, not the ranking.
    if jspace_best > 0.5 + _MIN_ABOVE_CHANCE:
        headline = "jspace_rises_under_tuned_penalty"
        print("THE J-SPACE NULL DOES NOT SURVIVE PENALTY TUNING.")
        print(f"  jspace reaches {jspace_best:.3f} at l2={jspace_best_l2:.0e}, against")
        print(f"  {table['jspace'][_FIXED]:.3f} at the fixed {_FIXED:.0e} used everywhere")
        print("  else. The null was an artifact of one regularisation choice.")
        print("  THIS OVERTURNS THE HEADLINE -- every J-space claim needs rerunning")
        print("  at a tuned penalty before anything is written.")
    else:
        headline = "jspace_null_survives_penalty_sweep"
        print("THE J-SPACE NULL SURVIVES THE PENALTY SWEEP.")
        print(f"  jspace peaks at {jspace_best:.3f} (l2={jspace_best_l2:.0e}) across four")
        print(f"  orders of magnitude, never clearing chance by {_MIN_ABOVE_CHANCE:.2f}.")
        print("  No regularisation story explains the null away.")

    # Then the anomaly, which is about the cross-source RANKING only.
    print()
    if best["residual"] > best["random_subspace"]:
        anomaly = "explained_by_regularisation"
        print("  ANOMALY EXPLAINED. At each source's own best penalty, residual")
        print(f"  ({best['residual']:.3f}) overtakes random_subspace "
              f"({best['random_subspace']:.3f}).")
        print("  The fixed penalty was over-regularising the 5120-dim residual and")
        print("  under-regularising the 16-dim projection. Report tuned numbers, and")
        print("  state that the fixed-penalty ranking was confounded.")
    else:
        anomaly = "unexplained"
        print("  ANOMALY UNEXPLAINED. random_subspace still leads residual at each")
        print(f"  source's best penalty ({best['random_subspace']:.3f} vs "
              f"{best['residual']:.3f}).")
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
