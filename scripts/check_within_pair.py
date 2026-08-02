#!/usr/bin/env python3
"""Is role encoded at all, just lexically? (diagnostic for the below-chance runs)

RQ1 and E4 both came back BELOW chance on the real model, with every fold
inverting:

    RQ1 final_token   jspace 0.32  orthogonal 0.28  residual 0.28
    E4  jspace        role_q 0.281  bag_q 0.263

Noise scatters around 0.5. Landing consistently below it means the probe learned
a rule on the training pairs that runs BACKWARDS on the held-out pair -- the
direction meaning "agent" for doctor/lawyer means "patient" for teacher/student.

Two very different situations produce that, and the experiments cannot separate
them because both use leave-one-pair-out:

  A. Role is encoded LEXICALLY, per concept pair. Each pair has its own role
     axis and they do not share a sign. Within a pair the probe would work;
     across pairs it inverts. Role is present, but not filler-general -- which
     is a claim about the representation, and directly on the proposal's
     question, since binding in the Smolensky sense requires role to be
     separable from filler.
  B. Role is not linearly readable at all. Within-pair decoding would also sit
     at chance, and the cross-pair inversion is an artifact of three folds.

This script runs the same probe WITHIN each concept pair (random split, no
leave-one-out) and reports both numbers side by side. Cheap: it reuses RQ1's
activations, so one pass over the stimuli, no edits, no fitted directions.

    within-pair HIGH + cross-pair BELOW chance  -> A: lexically entangled role
    within-pair CHANCE                          -> B: no linear role signal

    python scripts/check_within_pair.py --config configs/default.yaml --limit 200
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jspace_binding.analysis.probes import PROBE_SOURCES, ProbeExample, fit_ridge_probe
from jspace_binding.config import Config
from jspace_binding.experiments.progress import track
from jspace_binding.model.factory import build_model, preflight_or_exit
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.types import InjectionSite, Position, Role


def _within_pair_accuracy(rows: list[ProbeExample], seed: int, n_splits: int = 5) -> float:
    """Mean held-out accuracy from random splits INSIDE one concept pair.

    Random splits rather than leave-one-out because the question here is
    whether a role axis exists at all for this pair -- generalisation ACROSS
    pairs is exactly what the main experiments already measure.
    """
    rng = np.random.default_rng(seed)
    features = np.asarray([r.features for r in rows], dtype=float)
    labels = np.asarray([1.0 if r.is_agent else -1.0 for r in rows], dtype=float)
    accuracies = []
    for _split in range(n_splits):
        order = rng.permutation(len(rows))
        cut = int(0.7 * len(rows))
        train, test = order[:cut], order[cut:]
        if len(np.unique(labels[train])) < 2 or len(test) == 0:
            continue
        # fit_ridge_probe returns one augmented vector of shape (d + 1,); the
        # bias is its last entry, so predictions need the ones column appended.
        weights = fit_ridge_probe(features[train], labels[train])
        augmented = np.hstack([features[test], np.ones((len(test), 1))])
        predicted = np.sign(augmented @ weights)
        accuracies.append(float(np.mean(predicted == labels[test])))
    return float(np.mean(accuracies)) if accuracies else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--limit", type=int, default=200, help="families to read")
    parser.add_argument(
        "--site", default=InjectionSite.FINAL_TOKEN.value,
        choices=[s.value for s in InjectionSite],
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config)
    preflight_or_exit(model)
    site = InjectionSite(args.site)
    families = generate_families(config)[: args.limit]

    # pair_id -> source -> rows
    by_pair: dict[str, dict[str, list[ProbeExample]]] = defaultdict(
        lambda: {s: [] for s in PROBE_SOURCES}
    )
    for family in track(families, f"within-pair {site.value}", total=len(families)):
        entity = family.concept_pair.entity
        pair_id = family.concept_pair.pair_id
        for role in Role:
            for position in Position:
                sentence = family.cell(role, position).sentence
                activations = model.probe_activation(sentence, entity, site)
                for source in PROBE_SOURCES:
                    by_pair[pair_id][source].append(
                        ProbeExample(
                            pair_id=pair_id,
                            is_agent=role is Role.AGENT,
                            features=tuple(float(x) for x in activations[source]),
                        )
                    )

    print(f"\nmodel={config.model.model_id}  site={site.value}  families={len(families)}")
    print("chance = 0.50. WITHIN-pair splits are random; the main experiments")
    print("use leave-one-pair-out, which is what inverts.\n")

    report: dict[str, object] = {"site": site.value, "n_families": len(families)}
    per_source: dict[str, dict[str, float]] = {}
    for source in PROBE_SOURCES:
        print(f"  {source}")
        scores = {}
        for pair_id, sources in sorted(by_pair.items()):
            accuracy = _within_pair_accuracy(sources[source], config.experiment.seed)
            scores[pair_id] = accuracy
            print(f"    {pair_id:24s} within-pair {accuracy:.3f}")
        mean = float(np.mean(list(scores.values())))
        scores["mean"] = mean
        per_source[source] = scores
        print(f"    {'MEAN':24s}             {mean:.3f}\n")

    report["within_pair"] = per_source
    jspace_mean = per_source["jspace"]["mean"]
    residual_mean = per_source["residual"]["mean"]
    best = max(jspace_mean, residual_mean)

    print("=" * 68)
    if best > 0.65:
        verdict = "lexically_entangled"
        print("ROLE IS PRESENT BUT LEXICALLY ENTANGLED.")
        print(f"  Within-pair decoding works ({best:.3f}) while cross-pair inverts.")
        print("  Each concept pair carries its own role axis and they do not share")
        print("  a sign. Role is encoded, but NOT filler-general -- which is a")
        print("  claim about the representation, not a null result.")
    elif best > 0.55:
        verdict = "weak_lexical"
        print("WEAK within-pair signal.")
        print(f"  {best:.3f} is above chance but not decisive at this n.")
    else:
        verdict = "no_linear_role"
        print("NO LINEAR ROLE SIGNAL, even within a pair.")
        print(f"  Within-pair sits at chance ({best:.3f}), so the cross-pair")
        print("  inversion is not lexical entanglement -- role is simply not")
        print("  linearly readable here. A linear null does not rule out a")
        print("  multiplicative binding code (proposal, Potential Limitations).")
    print("=" * 68)

    report["verdict"] = verdict
    out = Path(config.paths.results) / "within_pair.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
