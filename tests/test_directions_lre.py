"""Tests for the LRE/LRC-style gradient estimator (directions.estimator).

Motivated by the Gemma-3-12B pilot: the diff-of-means push produced no directed
effect at any coefficient. diff-of-means fits the direction that SEPARATES
roles; the gradient estimator fits the direction the readout RESPONDS to
(Chanin et al. 2023). These tests pin the estimator's arithmetic and its
end-to-end path through fit_all and the dummy backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from jspace_binding.directions.fit import (
    FittedDirections,
    fit_all,
    fit_gradient_direction,
    gradient_bootstrap_stability,
    load_directions,
    random_sign_direction,
    save_directions,
)
from jspace_binding.model.dummy import DummyModel
from jspace_binding.types import InjectionSite


def _coherent_rows(rng: np.random.Generator, n: int = 40, d: int = 16) -> np.ndarray:
    axis = np.zeros(d)
    axis[0] = 1.0
    return axis + rng.normal(0.0, 0.3, size=(n, d))


def test_gradient_direction_is_unit_pooled_mean() -> None:
    rng = np.random.default_rng(0)
    rows = _coherent_rows(rng)
    direction = fit_gradient_direction(rows)
    assert direction.shape == (16,)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    expected = rows.mean(axis=0)
    assert np.allclose(direction, expected / np.linalg.norm(expected))


def test_gradient_direction_rejects_cancelling_rows() -> None:
    rows = np.array([[1.0, 0.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match="cancel"):
        fit_gradient_direction(rows)


def test_gradient_stability_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(1)
    coherent = gradient_bootstrap_stability(_coherent_rows(rng), n_resamples=100)
    noise = gradient_bootstrap_stability(rng.normal(size=(40, 16)), n_resamples=100)
    assert coherent > 0.95
    assert noise < coherent - 0.2


def test_random_sign_control_is_deterministic_and_collapses_norm() -> None:
    """Mirrors shuffled_label_direction's documented caveat: when the signal is
    strong, chance sign imbalance (~1/sqrt(n)) leaves the unit-normalized
    control partially aligned with the real axis, so cosine is NOT the
    diagnostic — the collapse of the RAW mean norm is."""
    rng = np.random.default_rng(2)
    rows = _coherent_rows(rng)
    control = random_sign_direction(rows, seed=0)
    assert np.allclose(control, random_sign_direction(rows, seed=0))
    real_norm = float(np.linalg.norm(rows.mean(axis=0)))
    signs = np.random.default_rng(0).choice((-1.0, 1.0), size=len(rows))
    scrambled_norm = float(np.linalg.norm((rows * signs[:, None]).mean(axis=0)))
    assert scrambled_norm < 0.5 * real_norm


def test_fit_all_gradient_estimator_and_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(3)
    activations = {
        "doctor": (_coherent_rows(rng, n=20), _coherent_rows(rng, n=20)),
        "teacher": (_coherent_rows(rng, n=20), _coherent_rows(rng, n=20)),
    }
    fitted = fit_all(
        activations, InjectionSite.FINAL_TOKEN, n_bootstrap=50, estimator="lre_gradient"
    )
    assert fitted.estimator == "lre_gradient"
    assert np.allclose(np.linalg.norm(fitted.fitted, axis=1), 1.0)
    save_directions(fitted, tmp_path)
    loaded = load_directions(tmp_path, InjectionSite.FINAL_TOKEN)
    assert loaded.estimator == "lre_gradient"
    assert np.allclose(loaded.fitted, fitted.fitted)

    with pytest.raises(ValueError, match="unknown estimator"):
        fit_all(activations, InjectionSite.FINAL_TOKEN, estimator="ridge")


def test_pre_estimator_npz_loads_as_diff_means(tmp_path) -> None:
    """npz files written before the estimator field existed must load."""
    d = FittedDirections(
        site=InjectionSite.FINAL_TOKEN,
        entities=("doctor",),
        fitted=np.eye(1, 4),
        shuffled=np.eye(1, 4),
        generic_loo=np.zeros((1, 4)),
        raw_norms=np.ones(1),
        stability=np.ones(1),
    )
    save_directions(d, tmp_path)
    # strip the estimator key to simulate an old file
    path = tmp_path / "directions_final_token.npz"
    data = dict(np.load(path, allow_pickle=False))
    data.pop("estimator")
    np.savez(path, **data)
    assert load_directions(tmp_path, InjectionSite.FINAL_TOKEN).estimator == "diff_means"


def test_dummy_gradient_recovers_planted_direction() -> None:
    """binding mode: pooled gradients must recover the planted direction with
    high stability and cosine ~1 against it; the sign is role-INdependent
    (unlike fitting_activation, whose sign flips with the exemplar's role)."""
    model = DummyModel(mode="binding")
    site = InjectionSite.ENTITY_TOKEN
    probe = "Who examined someone? Answer: The"
    rows = [
        model.fitting_gradient(
            f"The doctor examined the lawyer on day {i}.", probe, "doctor", "lawyer", site
        )
        for i in range(24)
    ]
    assert model.fitting_gradient(
        "The doctor examined the lawyer on day 0.", probe, "doctor", "lawyer", site
    ) == rows[0]  # deterministic
    direction = fit_gradient_direction(np.asarray(rows))
    planted = np.asarray(model._planted_direction("doctor", site))  # noqa: SLF001
    cosine = float(direction @ planted / np.linalg.norm(planted))
    assert cosine > 0.9
    assert gradient_bootstrap_stability(np.asarray(rows), n_resamples=100) > 0.9
