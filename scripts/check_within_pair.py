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


def _within_pair_accuracy(
    rows: list[ProbeExample], families: list[str], seed: int, n_splits: int = 5
) -> float:
    """Mean held-out accuracy from splits INSIDE one concept pair, held out BY
    FAMILY.

    Splitting by family, not at random over rows. A family's four cells share
    both entities and the verb and differ only in role and surface order, so a
    random row split puts near-duplicates on both sides and the probe can score
    high by recognising the family rather than reading the role. Holding whole
    families out removes that: the probe must generalise to sentences it has
    never seen, while still staying inside one concept pair -- which is the
    contrast this script exists to draw against the main experiments'
    leave-one-PAIR-out.
    """
    rng = np.random.default_rng(seed)
    features = np.asarray([r.features for r in rows], dtype=float)
    labels = np.asarray([1.0 if r.is_agent else -1.0 for r in rows], dtype=float)
    family_ids = np.asarray(families)
    unique = np.unique(family_ids)
    if len(unique) < 4:
        return float("nan")  # too few families to hold any out meaningfully

    accuracies = []
    for _split in range(n_splits):
        held_out = rng.choice(unique, size=max(1, len(unique) // 4), replace=False)
        test_mask = np.isin(family_ids, held_out)
        train, test = ~test_mask, test_mask
        if len(np.unique(labels[train])) < 2 or not test.any():
            continue
        # fit_ridge_probe returns one augmented vector of shape (d + 1,); the
        # bias is its last entry, so predictions need the ones column appended.
        weights = fit_ridge_probe(features[train], labels[train])
        augmented = np.hstack([features[test], np.ones((test.sum(), 1))])
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
    # Round-robin by concept pair, NOT a prefix. generate_families emits all of
    # one pair's families before the next, so `[:limit]` silently reduced the
    # first real run to a SINGLE pair -- and this script exists precisely to
    # compare pairs. Interleaving keeps every pair represented at any limit.
    all_families = generate_families(config)
    by_pair_all: dict[str, list] = defaultdict(list)
    for family in all_families:
        by_pair_all[family.concept_pair.pair_id].append(family)
    families = [
        family
        for group in zip(*by_pair_all.values(), strict=False)
        for family in group
    ][: args.limit]
    covered = {f.concept_pair.pair_id for f in families}
    print(f"sampling {len(families)} families across {len(covered)} pairs: {sorted(covered)}")

    # pair_id -> source -> rows
    by_pair: dict[str, dict[str, list[ProbeExample]]] = defaultdict(
        lambda: {s: [] for s in PROBE_SOURCES}
    )
    family_of: dict[str, list[str]] = defaultdict(list)  # pair_id -> per-row family id
    for family in track(families, f"within-pair {site.value}", total=len(families)):
        entity = family.concept_pair.entity
        pair_id = family.concept_pair.pair_id
        for role in Role:
            for position in Position:
                sentence = family.cell(role, position).sentence
                activations = model.probe_activation(sentence, entity, site)
                family_of[pair_id].append(family.family_id)
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
            accuracy = _within_pair_accuracy(
                sources[source], family_of[pair_id], config.experiment.seed
            )
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
    n_pairs = len(covered)
    if best > 0.65:
        verdict = "present_not_filler_general"
        print("ROLE IS PRESENT, BUT NOT FILLER-GENERAL.")
        print(f"  Within-pair decoding works ({best:.3f}) while cross-pair inverts.")
        print("  Role is linearly encoded, but the axis does not transfer across")
        print("  concept pairs -- so it is not separable from the filler, which is")
        print("  what role-filler binding requires. A claim about the")
        print("  representation, not a null result.")
        if per_source["jspace"]["mean"] < per_source["random_subspace"]["mean"] + 0.05:
            print()
            print("  AND: jspace does no better than a random subspace of equal rank")
            print(f"  ({per_source['jspace']['mean']:.3f} vs "
                  f"{per_source['random_subspace']['mean']:.3f}), while the orthogonal")
            print(f"  remainder reaches {per_source['orthogonal']['mean']:.3f}. The role")
            print("  signal is in the residual stream but NOT in the workspace.")
        if n_pairs < 2:
            print()
            print(f"  CAVEAT: only {n_pairs} concept pair covered. 'Does not transfer")
            print("  ACROSS pairs' cannot be shown from one pair -- raise --limit.")
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
