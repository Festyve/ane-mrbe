"""Sanity check for fitted role-directions.

Complements the proposal's Datasets §1 pilot check (the bootstrap-stability
resample, which lives in directions.fit) with two further questions — both
cheap, both a precondition for trusting any binding score built on a
direction:

1. *Does the fitted direction linearly separate role on HELD-OUT sentences?*
   Project held-out agent/patient J-space activations onto the entity's fitted
   unit direction and score the 1-D separation. The direction was fit on the
   training split only, so this is genuine generalization, not a memorized
   split. AUC is the headline (threshold-free: the probability a random agent
   sentence projects above a random patient one); accuracy at a mean-midpoint
   threshold is the "2-line classifier" companion.

2. *Are the per-entity directions related but distinct, or the same vector?*
   Pairwise cosine between the fitted directions. Near 1 across the board means
   the role axis is filler-general (one direction would do); moderate positive
   cosine means a shared role component plus entity-specific structure; near 0
   means entirely entity-specific axes.

Pure numpy — no torch, no matplotlib at import — so it runs anywhere and stays
unit-testable. The optional scatter figure lives in :func:`scatter_projections`,
which imports matplotlib lazily and degrades to a no-op if it is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SeparationResult:
    """How well a fitted direction separates held-out agent/patient projections."""

    entity: str
    site: str
    n_agent: int
    n_patient: int
    auc: float  # P(agent proj > patient proj); 0.5 = chance, 1.0 = perfect
    accuracy: float  # at the mean-midpoint threshold
    threshold: float  # midpoint of the two class means (in projection units)
    agent_mean: float
    patient_mean: float
    cohens_d: float  # standardized separation of the two projection clouds

    @property
    def separates(self) -> bool:
        """Heuristic pass flag: clearly-above-chance ranking. Diagnostic only —
        the numbers travel in the summary regardless."""
        return self.auc >= 0.75


def project(activations: np.ndarray, unit_direction: np.ndarray) -> np.ndarray:
    """1-D projections of each activation row onto ``unit_direction``.

    ``unit_direction`` is assumed unit-norm (the fit stores it so); a non-unit
    vector only rescales every projection and leaves AUC/accuracy unchanged.
    """
    activations = np.atleast_2d(np.asarray(activations, dtype=float))
    unit_direction = np.asarray(unit_direction, dtype=float)
    if activations.shape[1] != unit_direction.shape[0]:
        raise ValueError(
            f"project: activation dim {activations.shape[1]} != direction dim "
            f"{unit_direction.shape[0]}"
        )
    return activations @ unit_direction


def _auc(agent_proj: np.ndarray, patient_proj: np.ndarray) -> float:
    """Mann-Whitney AUC: fraction of (agent, patient) pairs the direction ranks
    correctly, ties counting as half. Threshold-free separation measure."""
    wins = 0.0
    for a in agent_proj:
        wins += float(np.sum(a > patient_proj)) + 0.5 * float(np.sum(a == patient_proj))
    return wins / (len(agent_proj) * len(patient_proj))


def separation_report(
    agent_proj: np.ndarray, patient_proj: np.ndarray, entity: str, site: str
) -> SeparationResult:
    """Score the 1-D separation of held-out agent vs patient projections.

    The classifier is the honest held-out one: threshold at the midpoint of the
    two class means, predict "agent" on the side of the higher agent mean (a
    fitted agent-minus-patient direction should put agents higher, but we orient
    by the data so a sign flip does not read as chance).
    """
    agent_proj = np.asarray(agent_proj, dtype=float).ravel()
    patient_proj = np.asarray(patient_proj, dtype=float).ravel()
    if len(agent_proj) == 0 or len(patient_proj) == 0:
        raise ValueError(f"separation_report[{entity}/{site}]: empty projection set")

    a_mean, p_mean = float(agent_proj.mean()), float(patient_proj.mean())
    threshold = 0.5 * (a_mean + p_mean)
    # Orient the decision so the class with the higher mean is "positive".
    sign = 1.0 if a_mean >= p_mean else -1.0
    correct = np.sum(sign * agent_proj > sign * threshold) + np.sum(
        sign * patient_proj <= sign * threshold
    )
    accuracy = float(correct) / (len(agent_proj) + len(patient_proj))

    pooled_var = 0.5 * (agent_proj.var(ddof=0) + patient_proj.var(ddof=0))
    cohens_d = 0.0 if pooled_var == 0 else abs(a_mean - p_mean) / float(np.sqrt(pooled_var))

    return SeparationResult(
        entity=entity,
        site=site,
        n_agent=len(agent_proj),
        n_patient=len(patient_proj),
        auc=_auc(agent_proj, patient_proj),
        accuracy=accuracy,
        threshold=threshold,
        agent_mean=a_mean,
        patient_mean=p_mean,
        cohens_d=cohens_d,
    )


def cosine_matrix(
    directions: dict[str, np.ndarray],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Pairwise cosine similarity between fitted directions.

    Returns the entity order and a symmetric matrix. Inputs need not be unit
    (each is normalized here); a zero vector raises rather than dividing by 0.
    """
    entities = tuple(directions)
    mat = np.stack([np.asarray(directions[e], dtype=float) for e in entities])
    norms = np.linalg.norm(mat, axis=1)
    if np.any(norms < 1e-12):
        zero = [e for e, n in zip(entities, norms) if n < 1e-12]
        raise ValueError(f"cosine_matrix: zero-norm direction(s) for {zero}")
    unit = mat / norms[:, None]
    return entities, unit @ unit.T


def compare_directions(directions: dict[str, np.ndarray]) -> dict[str, object]:
    """Cosine comparison of per-entity directions with a plain-language verdict.

    off_diagonal is every distinct pair's cosine; the verdict buckets the mean
    off-diagonal cosine into same-vector / shared-plus-specific / distinct so
    the writeup can state the qualitative finding directly.
    """
    if len(directions) < 2:
        raise ValueError("compare_directions needs >= 2 directions")
    entities, mat = cosine_matrix(directions)
    pairs = {
        f"{entities[i]}~{entities[j]}": float(mat[i, j])
        for i in range(len(entities))
        for j in range(i + 1, len(entities))
    }
    mean_cos = float(np.mean(list(pairs.values())))
    if mean_cos >= 0.9:
        verdict = "near-identical (role axis is filler-general)"
    elif mean_cos >= 0.3:
        verdict = "related but distinct (shared role component + entity-specific structure)"
    else:
        verdict = "essentially distinct (entity-specific axes)"
    return {
        "entities": list(entities),
        "cosine_matrix": mat.tolist(),
        "pairs": pairs,
        "mean_off_diagonal_cosine": mean_cos,
        "verdict": verdict,
    }


def scatter_projections(
    projections: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
) -> Path | None:
    """Strip/scatter plot of held-out projections, agent vs patient, per entity.

    ``projections`` maps a label (e.g. "doctor@final_token") to
    (agent_proj, patient_proj). Returns the written path, or None if matplotlib
    is unavailable (broken/absent build) — the numeric summary never depends on
    the figure, so a missing plotting stack degrades gracefully.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, OSError):
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(projections)
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(labels) + 1.5))
    rng = np.random.default_rng(0)
    for row, label in enumerate(labels):
        agent_proj, patient_proj = projections[label]
        for proj, color, name in (
            (agent_proj, "tab:blue", "agent"),
            (patient_proj, "tab:orange", "patient"),
        ):
            proj = np.asarray(proj, dtype=float).ravel()
            jitter = row + 0.12 * rng.standard_normal(len(proj))
            ax.scatter(proj, jitter, s=18, alpha=0.7, color=color,
                       label=name if row == 0 else None)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("projection onto fitted role-direction")
    ax.set_title("Held-out role separation along the fitted direction")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
