"""RQ1 linear probes: is role decodable, and from where? Pure numpy.

The proposal's warm-up analysis (§4): train a linear probe to predict the
target entity's role (agent vs patient) from three activation sources — the
J-space component, the orthogonal remainder, and the full residual stream —
and report control-task selectivity (Hewitt & Liang, 2019) alongside raw
accuracy, so "information present" is distinguishable from "probe memorized
the stimuli".

Interpretation guardrails (proposal, Potential Limitations): a positive
result shows decodability, not use; a negative result is nearly
uninformative, because tensor-product-style binding may be multiplicatively
encoded and invisible to a linear readout (Smolensky, 1990).

Probe: ridge-regularized least squares to +/-1 labels with a bias term,
closed form, deterministic — no sklearn dependency, no iterative training.
Splits are BY LEXICAL PAIR (leave-one-pair-out), so no profession noun
appears on both sides of a split (proposal, Datasets / Training data).

Control task: each example gets a deterministic pseudo-random +/-1 label
(seeded, content-independent of role). A probe that can hit high accuracy on
these arbitrary labels is fitting stimulus identity, not reading role;
selectivity = task accuracy - control accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROBE_SOURCES: tuple[str, ...] = ("jspace", "orthogonal", "residual")


@dataclass(frozen=True)
class ProbeExample:
    """One cached activation with its labels and split group."""

    pair_id: str  # split group: leave-one-pair-out
    is_agent: bool  # task label
    features: tuple[float, ...]  # one activation source's vector


def fit_ridge_probe(
    features: np.ndarray, labels: np.ndarray, l2: float = 1e-2
) -> np.ndarray:
    """Closed-form ridge regression to +/-1 labels, bias included.

    Returns w of shape (d + 1,); predictions are sign(X_aug @ w). The bias
    column is appended, not penalized differently — at these scales the
    distinction is immaterial and the closed form stays one line.
    """
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("fit_ridge_probe: features must be (n, d) aligned with labels")
    augmented = np.hstack([features, np.ones((len(features), 1))])
    gram = augmented.T @ augmented + l2 * np.eye(augmented.shape[1])
    return np.linalg.solve(gram, augmented.T @ labels.astype(float))


def probe_accuracy(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of sign-correct predictions; ties (raw 0) count as errors."""
    augmented = np.hstack([features, np.ones((len(features), 1))])
    predictions = np.sign(augmented @ weights)
    return float(np.mean(predictions == labels))


def _labels(examples: list[ProbeExample]) -> np.ndarray:
    return np.asarray([1.0 if ex.is_agent else -1.0 for ex in examples])


def _control_labels(examples: list[ProbeExample], seed: int) -> np.ndarray:
    """Deterministic pseudo-random +/-1 per example, independent of role.

    Derived from a content hash of the example (sha256, NOT Python's salted
    hash()), so the control label of an example is stable across splits and
    across runs (Hewitt & Liang's control task assigns labels to items, not
    to occurrences).
    """
    import hashlib

    out = np.empty(len(examples))
    for i, ex in enumerate(examples):
        key = f"{seed}\x1f{ex.pair_id}\x1f{','.join(f'{x:.9g}' for x in ex.features)}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        out[i] = 1.0 if digest[0] % 2 == 0 else -1.0
    return out


@dataclass(frozen=True)
class ProbeReport:
    """Leave-one-pair-out results for one activation source at one site."""

    accuracy: float  # mean held-out task accuracy over folds
    control_accuracy: float  # mean held-out control-task accuracy over folds
    n_examples: int
    n_folds: int
    fold_ids: tuple[str, ...] = ()  # held-out pair_id per fold, aligned with below
    fold_accuracies: tuple[float, ...] = ()
    control_fold_accuracies: tuple[float, ...] = ()

    @property
    def selectivity(self) -> float:
        return self.accuracy - self.control_accuracy

    @property
    def fold_spread(self) -> float:
        """max - min held-out accuracy across folds; 0.0 if not recorded.

        Large spread means the mean is not summarizing a stable effect.
        """
        if not self.fold_accuracies:
            return 0.0
        return float(max(self.fold_accuracies) - min(self.fold_accuracies))

    @property
    def inverting_folds(self) -> tuple[str, ...]:
        """Held-out pairs the probe scored materially BELOW chance on.

        Such a fold is not weak evidence of absence: it means the rule learned
        on the training pairs runs backwards on this one, which is a statement
        about cross-pair transfer, not about whether role is encoded. Named so
        the offending lexical pair is identifiable without a rerun.
        """
        return tuple(
            pair
            for pair, acc in zip(self.fold_ids, self.fold_accuracies, strict=False)
            if acc < 0.4
        )


def leave_one_pair_out(
    examples: list[ProbeExample], l2: float = 1e-2, control_seed: int = 0
) -> ProbeReport:
    """Cross-validated probe accuracy + control accuracy, split by pair_id.

    Each fold holds out every example of one concept pair; the probe never
    sees the held-out professions during training, so above-chance accuracy
    requires role information that generalizes across lexical items.

    Both the mean and the per-fold accuracies are returned. Read the folds
    first: with one fold per concept pair there are as few as three of them,
    and the mean alone cannot distinguish "transfers weakly everywhere" from
    "transfers on most pairs and inverts on one" (see ProbeReport).
    """
    pair_ids = sorted({ex.pair_id for ex in examples})
    if len(pair_ids) < 2:
        raise ValueError(
            f"leave_one_pair_out: need >= 2 concept pairs to split by pair, got {pair_ids}"
        )
    features = np.asarray([ex.features for ex in examples])
    task = _labels(examples)
    control = _control_labels(examples, seed=control_seed)
    groups = np.asarray([ex.pair_id for ex in examples])

    task_accs, control_accs = [], []
    for held_out in pair_ids:
        test = groups == held_out
        train = ~test
        w_task = fit_ridge_probe(features[train], task[train], l2=l2)
        task_accs.append(probe_accuracy(w_task, features[test], task[test]))
        w_control = fit_ridge_probe(features[train], control[train], l2=l2)
        control_accs.append(probe_accuracy(w_control, features[test], control[test]))
    return ProbeReport(
        accuracy=float(np.mean(task_accs)),
        control_accuracy=float(np.mean(control_accs)),
        n_examples=len(examples),
        n_folds=len(pair_ids),
        fold_ids=tuple(pair_ids),
        fold_accuracies=tuple(float(a) for a in task_accs),
        control_fold_accuracies=tuple(float(a) for a in control_accs),
    )
