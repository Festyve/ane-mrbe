"""Closed-form role-direction fitting over cached J-space activations.

Pure numpy; the backend supplies activations already projected into the J-space
subspace (model.fitting_activation) and everything here is arithmetic. The fit
is per injection site,

    r_entity = mean(activation | entity = agent) - mean(activation | entity = patient)

stored as a UNIT vector plus its raw norm, so a push of coefficient * unit(r)
makes the RANDOM_DIRECTION control norm-matched by construction.

Three variants per (entity, site): `fitted` (the real direction), `shuffled`
(refit on shuffled labels — the direction-overfitting control), and
`generic_loo` (leave-one-entity-out mean of the others, which tests whether the
role structure is filler-general). Bootstrap stability reports the mean cosine
between resampled fits and the full fit; a low value means the corpus is too
small or carries no role signal, and the fit script warns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jspace_binding.types import InjectionSite


@dataclass(frozen=True)
class FittedDirections:
    """All directions fitted at one injection site.

    entities orders the rows of every matrix; all directions are unit-norm
    rows. raw_norms holds |mean_agent - mean_patient| before normalization
    (diagnostic only — pushes always use unit directions).
    """

    site: InjectionSite
    entities: tuple[str, ...]
    fitted: np.ndarray  # (n_entities, d) unit rows
    shuffled: np.ndarray  # (n_entities, d) unit rows
    generic_loo: np.ndarray  # (n_entities, d) unit rows
    raw_norms: np.ndarray  # (n_entities,)
    stability: np.ndarray  # (n_entities,) mean bootstrap cosine vs full fit
    estimator: str = "diff_means"  # which estimator produced `fitted` (see fit_all)

    def index(self, entity: str) -> int:
        try:
            return self.entities.index(entity)
        except ValueError:
            raise KeyError(
                f"no direction fitted for entity {entity!r} at site {self.site.value}; "
                f"fitted entities: {', '.join(self.entities)}"
            ) from None

    def direction(self, entity: str, variant: str = "fitted") -> np.ndarray:
        matrix = {
            "fitted": self.fitted,
            "shuffled": self.shuffled,
            "generic_loo": self.generic_loo,
        }.get(variant)
        if matrix is None:
            raise ValueError(f"unknown direction variant {variant!r}")
        row = matrix[self.index(entity)]
        if variant == "generic_loo" and float(np.linalg.norm(row)) < 1e-12:
            raise ValueError(
                f"generic_loo direction unavailable for {entity!r}: the corpus was "
                "fitted from a single entity (leave-one-out needs >= 2)"
            )
        return row


def fit_role_direction(agent_acts: np.ndarray, patient_acts: np.ndarray) -> np.ndarray:
    """Unit difference-of-means direction: unit(mean(agent) - mean(patient)).

    Raises on a (near-)zero difference — a degenerate fit would make the push
    a silent no-op and every downstream score meaningless.
    """
    agent_acts = np.asarray(agent_acts, dtype=float)
    patient_acts = np.asarray(patient_acts, dtype=float)
    if agent_acts.ndim != 2 or patient_acts.ndim != 2:
        raise ValueError("fit_role_direction: activations must be 2-D (n_examples, d)")
    if agent_acts.shape[1] != patient_acts.shape[1]:
        raise ValueError(
            f"fit_role_direction: dimension mismatch "
            f"({agent_acts.shape[1]} vs {patient_acts.shape[1]})"
        )
    raw = agent_acts.mean(axis=0) - patient_acts.mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if norm < 1e-12:
        raise ValueError("fit_role_direction: agent/patient means coincide (zero direction)")
    return raw / norm




def fit_gradient_direction(rows: np.ndarray) -> np.ndarray:
    """Unit mean of per-exemplar readout gradients (LRE-style estimator,
    Chanin et al. 2023, arXiv:2311.08968).

    Each row is d(z_entity - z_other)/dh at the site token, pooled over both
    roles' exemplars — the contrast lives inside each row, so there is no class
    difference to subtract. diff-of-means finds the direction that SEPARATES
    roles; this finds the one that STEERS the readout, and they need not agree.
    """
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2:
        raise ValueError("fit_gradient_direction: rows must be 2-D (n_examples, d)")
    raw = rows.mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if norm < 1e-12:
        raise ValueError("fit_gradient_direction: exemplar gradients cancel (zero direction)")
    return raw / norm


def gradient_bootstrap_stability(
    rows: np.ndarray,
    n_resamples: int = 200,
    seed: int = 0,
) -> float:
    """bootstrap_stability's analog for the pooled-gradient estimator."""
    rows = np.asarray(rows, dtype=float)
    reference = fit_gradient_direction(rows)
    rng = np.random.default_rng(seed)
    cosines = np.zeros(n_resamples)
    for b in range(n_resamples):
        raw = rows[rng.integers(0, len(rows), size=len(rows))].mean(axis=0)
        norm = float(np.linalg.norm(raw))
        cosines[b] = 0.0 if norm < 1e-12 else float(raw @ reference) / norm
    return float(cosines.mean())


def random_sign_direction(rows: np.ndarray, seed: int = 0) -> np.ndarray:
    """shuffled_label_direction's analog for the gradient estimator.

    Labels play no role in a pooled mean, so shuffling them is a no-op there;
    the null that destroys the coherent signal while keeping each exemplar's
    magnitude structure is scrambling the SIGNS instead.
    """
    rows = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=len(rows))
    raw = (rows * signs[:, None]).mean(axis=0)
    norm = float(np.linalg.norm(raw))
    if norm < 1e-12:
        raise ValueError("random_sign_direction: scrambled gradients cancel (zero direction)")
    return raw / norm


def bootstrap_stability(
    agent_acts: np.ndarray,
    patient_acts: np.ndarray,
    n_resamples: int = 200,
    seed: int = 0,
) -> float:
    """Mean cosine between bootstrap-refit directions and the full-fit direction.

    Resamples the two classes independently with replacement. Near 1 means the
    direction is stable; near 0 means the corpus is too small or noisy to pin
    one down. Degenerate resamples count as cosine 0.
    """
    agent_acts = np.asarray(agent_acts, dtype=float)
    patient_acts = np.asarray(patient_acts, dtype=float)
    reference = fit_role_direction(agent_acts, patient_acts)
    rng = np.random.default_rng(seed)
    cosines = np.zeros(n_resamples)
    for b in range(n_resamples):
        a_idx = rng.integers(0, len(agent_acts), size=len(agent_acts))
        p_idx = rng.integers(0, len(patient_acts), size=len(patient_acts))
        raw = agent_acts[a_idx].mean(axis=0) - patient_acts[p_idx].mean(axis=0)
        norm = float(np.linalg.norm(raw))
        cosines[b] = 0.0 if norm < 1e-12 else float(raw @ reference) / norm
    return float(cosines.mean())


def shuffled_label_direction(
    agent_acts: np.ndarray,
    patient_acts: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Refit with agent/patient labels shuffled: the direction-overfitting control.

    Pools all rows, deals them back into two groups of the original sizes, and
    runs the same difference-of-means. Carries no role signal on average, but
    under a strong true signal a chance label imbalance leaves a single
    shuffled direction partially aligned with the role axis — the collapse of
    its RAW norm is the cleaner per-shuffle diagnostic. As a null-band control
    the bias is conservative: it can only widen the band, never shrink it.
    """
    agent_acts = np.asarray(agent_acts, dtype=float)
    patient_acts = np.asarray(patient_acts, dtype=float)
    pooled = np.concatenate([agent_acts, patient_acts], axis=0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pooled))
    fake_agent = pooled[order[: len(agent_acts)]]
    fake_patient = pooled[order[len(agent_acts) :]]
    return fit_role_direction(fake_agent, fake_patient)


def generic_loo_direction(
    fitted_by_entity: dict[str, np.ndarray], held_out: str
) -> np.ndarray:
    """Leave-one-out generic role direction: unit mean of the OTHER entities'
    fitted unit directions (proposal's filler-general robustness variant)."""
    others = [v for k, v in fitted_by_entity.items() if k != held_out]
    if not others:
        raise ValueError(f"generic_loo_direction: no entities besides {held_out!r}")
    mean = np.mean(np.stack(others), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-12:
        raise ValueError("generic_loo_direction: other entities' directions cancel out")
    return mean / norm


def fit_all(
    activations: dict[str, tuple[np.ndarray, np.ndarray]],
    site: InjectionSite,
    n_bootstrap: int = 200,
    seed: int = 0,
    estimator: str = "diff_means",
) -> FittedDirections:
    """Fit every entity's direction plus both control variants at one site.

    activations maps entity -> (agent_acts, patient_acts). Shuffle seeds are
    offset per entity. With a single-entity corpus the leave-one-out generic
    variant is undefined; its rows are stored as zeros and direction() raises.

    estimator "diff_means" fits unit(mean(agent) - mean(patient));
    "lre_gradient" treats the rows as readout gradients and fits their unit
    pooled mean, with a sign-scramble null in place of the label shuffle.
    """
    if estimator not in ("diff_means", "lre_gradient"):
        raise ValueError(f"unknown estimator {estimator!r}")
    entities = tuple(sorted(activations))
    fitted: dict[str, np.ndarray] = {}
    shuffled_rows, norms, stabilities = [], [], []
    for offset, entity in enumerate(entities):
        agent_acts, patient_acts = activations[entity]
        if estimator == "lre_gradient":
            pooled = np.concatenate(
                [np.asarray(agent_acts, float), np.asarray(patient_acts, float)], axis=0
            )
            norms.append(float(np.linalg.norm(pooled.mean(axis=0))))
            fitted[entity] = fit_gradient_direction(pooled)
            shuffled_rows.append(random_sign_direction(pooled, seed=seed + offset))
            stabilities.append(
                gradient_bootstrap_stability(pooled, n_resamples=n_bootstrap, seed=seed)
            )
            continue
        raw = np.asarray(agent_acts, float).mean(axis=0) - np.asarray(patient_acts, float).mean(
            axis=0
        )
        norms.append(float(np.linalg.norm(raw)))
        fitted[entity] = fit_role_direction(agent_acts, patient_acts)
        shuffled_rows.append(shuffled_label_direction(agent_acts, patient_acts, seed=seed + offset))
        stabilities.append(
            bootstrap_stability(agent_acts, patient_acts, n_resamples=n_bootstrap, seed=seed)
        )
    if len(entities) >= 2:
        generic = np.stack([generic_loo_direction(fitted, e) for e in entities])
    else:
        generic = np.zeros_like(np.stack([fitted[e] for e in entities]))
    return FittedDirections(
        site=site,
        entities=entities,
        fitted=np.stack([fitted[e] for e in entities]),
        shuffled=np.stack(shuffled_rows),
        generic_loo=generic,
        raw_norms=np.asarray(norms),
        stability=np.asarray(stabilities),
        estimator=estimator,
    )


def save_directions(directions: FittedDirections, directory: str | Path) -> Path:
    """Write directions_{site}.npz (+ a human-readable JSON summary)."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"directions_{directions.site.value}.npz"
    np.savez(
        path,
        site=directions.site.value,
        entities=np.asarray(directions.entities),
        fitted=directions.fitted,
        shuffled=directions.shuffled,
        generic_loo=directions.generic_loo,
        raw_norms=directions.raw_norms,
        stability=directions.stability,
        estimator=directions.estimator,
    )
    summary = {
        "site": directions.site.value,
        "estimator": directions.estimator,
        "entities": list(directions.entities),
        "raw_norms": [float(x) for x in directions.raw_norms],
        "stability": [float(x) for x in directions.stability],
    }
    (out_dir / f"directions_{directions.site.value}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_directions(directory: str | Path, site: InjectionSite) -> FittedDirections:
    path = Path(directory) / f"directions_{site.value}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no fitted directions at {path}; run scripts/fit_directions.py first"
        )
    data = np.load(path, allow_pickle=False)
    return FittedDirections(
        site=InjectionSite(str(data["site"])),
        entities=tuple(str(e) for e in data["entities"]),
        fitted=data["fitted"],
        shuffled=data["shuffled"],
        generic_loo=data["generic_loo"],
        raw_norms=data["raw_norms"],
        stability=data["stability"],
        # npz files written before the estimator existed are all diff-of-means
        estimator=str(data["estimator"]) if "estimator" in data.files else "diff_means",
    )
