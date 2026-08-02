#!/usr/bin/env python3
"""Is the J-space null an artifact of using LINEAR probes? (Potential Limitations)

The proposal concedes that a linear null does not rule out a multiplicative /
tensor-product binding code (Smolensky 1990): if role and filler are combined by
an outer product, no linear readout can recover role, and every result in this
project would be a statement about our instrument rather than about the model.
That concession is currently unaddressed empirically, and a reviewer will treat
an unaddressed escape hatch as the whole paper's escape hatch.

This closes it. Same cached activations, same leave-one-pair-out split, three
probe families of increasing expressivity:

  linear   ridge on the raw activation.  The existing RQ1 probe; baseline.
  quad     ridge on random PAIRWISE PRODUCTS x_i * x_j.  Aimed squarely at the
           multiplicative hypothesis: a tensor-product code is exactly a
           bilinear form in the activation, so a quadratic probe can read one
           where a linear probe provably cannot.
  rff      ridge on random ReLU features (a random one-hidden-layer net with a
           closed-form output layer).  General nonlinearity, in case the code is
           neither linear nor cleanly bilinear.

Random features rather than a trained MLP deliberately: closed-form via the
project's own fit_ridge_probe, no SGD, no learning-rate or early-stopping
choices that a reviewer can call p-hacking, and deterministic given the seed.
The expressivity argument is the standard one -- random ReLU features are a
universal approximator as the feature count grows.

READING THE OUTPUT. The decisive contrast is not "does nonlinear beat linear"
in general (it usually will, from extra capacity alone) but the SOURCE pattern:

  nonlinear lifts residual but NOT jspace  -> the null is about J-SPACE, not
                                             about linearity. Escape closed.
  nonlinear lifts jspace to residual level -> role IS in J-space, multiplicatively
                                             encoded. The headline flips, and
                                             that is a bigger finding than the null.

random_subspace is the capacity control throughout: extra expressivity helps
every source somewhat, so a jspace gain only counts if it exceeds the gain a
rank-matched random subspace gets for free.

    python scripts/check_nonlinear_probe.py --config configs/default.yaml --limit 200
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

_N_FEATURES = 2048  # random features per nonlinear map
PROBE_KINDS: tuple[str, ...] = ("linear", "quad", "rff")


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score using TRAIN statistics only.

    Fitted on train alone because the held-out pair must stay held out: using
    pooled statistics leaks the test distribution into the feature map, which
    on a leave-one-pair-out split is exactly the generalisation being measured.
    """
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _features(
    kind: str, train: np.ndarray, test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw activations to the probe's feature space.

    The random projections are drawn from `seed` alone, so they are identical
    across sources and folds -- a jspace-vs-residual difference cannot come
    from a luckier draw.
    """
    if kind == "linear":
        return train, test

    rng = np.random.default_rng(seed)
    train, test = _standardise(train, test)
    d = train.shape[1]

    if kind == "quad":
        # Standardised x CONCATENATED with random pairwise products x_i * x_j.
        #
        # The products are the point -- a tensor-product (role (x) filler) code
        # is a bilinear form, so its role component is linear in exactly these
        # features while being invisible to a probe on x itself. But the raw
        # terms have to come along, or the probe is not a SUPERSET of the
        # linear one and a null tells you nothing about linearity.
        #
        # Caught by the dummy: on `binding` mode, where role is planted
        # LINEARLY in jspace, products-only scored exactly 0.500 (chance) while
        # the linear probe scored 0.990. Products alone discard the linear
        # term, so a products-only null is uninformative about whether the
        # linear null was an artifact -- which is the entire question here.
        i = rng.integers(0, d, size=_N_FEATURES)
        j = rng.integers(0, d, size=_N_FEATURES)
        return (
            np.hstack([train, train[:, i] * train[:, j]]),
            np.hstack([test, test[:, i] * test[:, j]]),
        )

    if kind == "rff":
        # Random ReLU features: a one-hidden-layer net with a random hidden
        # layer and a closed-form (ridge) output layer.
        weights = rng.standard_normal((d, _N_FEATURES)) / np.sqrt(d)
        bias = rng.standard_normal(_N_FEATURES) * 0.1
        return np.maximum(0.0, train @ weights + bias), np.maximum(0.0, test @ weights + bias)

    raise ValueError(f"unknown probe kind {kind!r}")


def _leave_one_pair_out(rows: list[ProbeExample], kind: str, seed: int) -> float:
    """Held-out accuracy, holding out one concept pair at a time.

    Same split as RQ1, so the numbers sit directly beside it.
    """
    features = np.asarray([r.features for r in rows], dtype=float)
    labels = np.asarray([1.0 if r.is_agent else -1.0 for r in rows], dtype=float)
    pair_ids = np.asarray([r.pair_id for r in rows])

    accuracies = []
    for held_out in np.unique(pair_ids):
        test_mask = pair_ids == held_out
        train_mask = ~test_mask
        if len(np.unique(labels[train_mask])) < 2 or not test_mask.any():
            continue
        train_x, test_x = _features(kind, features[train_mask], features[test_mask], seed)
        weights = fit_ridge_probe(train_x, labels[train_mask])
        augmented = np.hstack([test_x, np.ones((len(test_x), 1))])
        predicted = np.sign(augmented @ weights)
        accuracies.append(float(np.mean(predicted == labels[test_mask])))
    return float(np.mean(accuracies)) if accuracies else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--site", default=InjectionSite.ENTITY_TOKEN.value,
        choices=[s.value for s in InjectionSite],
        help="default entity_token: the site where role IS linearly decodable, "
             "so a jspace gain there cannot be dismissed as 'no signal anywhere'",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config)
    preflight_or_exit(model)
    site = InjectionSite(args.site)

    # Round-robin across pairs so --limit never silently covers one pair.
    grouped: dict[str, list] = defaultdict(list)
    for family in generate_families(config):
        grouped[family.concept_pair.pair_id].append(family)
    families = [f for group in zip(*grouped.values(), strict=False) for f in group][: args.limit]
    print(f"site={site.value}  families={len(families)}  "
          f"pairs={len({f.concept_pair.pair_id for f in families})}")

    rows: dict[str, list[ProbeExample]] = {s: [] for s in PROBE_SOURCES}
    for family in track(families, f"nonlinear {site.value}", total=len(families)):
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

    seed = config.experiment.seed
    table = {
        source: {kind: _leave_one_pair_out(rows[source], kind, seed) for kind in PROBE_KINDS}
        for source in PROBE_SOURCES
    }

    print(f"\nleave-one-pair-out accuracy, chance = 0.50   (site={site.value})\n")
    print(f"  {'source':18s}" + "".join(f"{k:>10s}" for k in PROBE_KINDS) + f"{'lift':>10s}")
    print("  " + "-" * 58)
    for source, scores in table.items():
        best_nonlinear = max(scores["quad"], scores["rff"])
        lift = best_nonlinear - scores["linear"]
        print(
            f"  {source:18s}"
            + "".join(f"{scores[k]:10.3f}" for k in PROBE_KINDS)
            + f"{lift:+10.3f}"
        )

    jspace_lift = max(table["jspace"]["quad"], table["jspace"]["rff"]) - table["jspace"]["linear"]
    control_lift = (
        max(table["random_subspace"]["quad"], table["random_subspace"]["rff"])
        - table["random_subspace"]["linear"]
    )
    jspace_best = max(table["jspace"][k] for k in PROBE_KINDS)
    residual_best = max(table["residual"][k] for k in PROBE_KINDS)

    print("\n" + "=" * 62)
    if jspace_best > residual_best - 0.1 and jspace_lift > control_lift + 0.05:
        verdict = "multiplicative_code_in_jspace"
        print("ROLE IS IN J-SPACE, NONLINEARLY ENCODED.")
        print(f"  jspace reaches {jspace_best:.3f} under a nonlinear probe (linear:")
        print(f"  {table['jspace']['linear']:.3f}), a lift of {jspace_lift:+.3f} against")
        print(f"  {control_lift:+.3f} for the rank-matched control. The linear null was")
        print("  an artifact of the readout. THIS OVERTURNS THE HEADLINE.")
    else:
        verdict = "null_survives_nonlinearity"
        print("THE J-SPACE NULL SURVIVES NONLINEAR PROBING.")
        print(f"  jspace tops out at {jspace_best:.3f} vs residual {residual_best:.3f};")
        print(f"  its nonlinear lift is {jspace_lift:+.3f} against {control_lift:+.3f} for a")
        print("  rank-matched random subspace. A quadratic probe -- which CAN read a")
        print("  tensor-product code -- finds no more role in J-space than a linear one.")
        print("  The multiplicative-encoding escape hatch is closed empirically.")
    print("=" * 62)

    out = Path(config.paths.results) / "nonlinear_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"site": site.value, "n_families": len(families), "accuracy": table,
             "jspace_lift": jspace_lift, "capacity_control_lift": control_lift,
             "verdict": verdict},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
