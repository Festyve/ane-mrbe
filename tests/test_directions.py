"""Contract tests for directions.fit: recovery, stability, controls, storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jspace_binding.directions.fit import (
    bootstrap_stability,
    fit_all,
    fit_role_direction,
    generic_loo_direction,
    load_directions,
    save_directions,
    shuffled_label_direction,
)
from jspace_binding.types import InjectionSite

D = 16
N = 40


def _planted(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(D)
    return raw / np.linalg.norm(raw)


def _role_activations(
    planted: np.ndarray, noise: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (agent, patient) activations: +/- planted + Gaussian noise."""
    rng = np.random.default_rng(seed)
    agent = planted + noise * rng.standard_normal((N, D))
    patient = -planted + noise * rng.standard_normal((N, D))
    return agent, patient


def test_fit_recovers_planted_direction() -> None:
    planted = _planted(0)
    agent, patient = _role_activations(planted, noise=0.3, seed=1)
    fitted = fit_role_direction(agent, patient)
    assert np.linalg.norm(fitted) == pytest.approx(1.0)
    assert float(fitted @ planted) > 0.97


def test_fit_rejects_degenerate_inputs() -> None:
    same = np.ones((4, D))
    with pytest.raises(ValueError, match="zero direction"):
        fit_role_direction(same, same.copy())
    with pytest.raises(ValueError, match="dimension mismatch"):
        fit_role_direction(np.ones((4, D)), np.ones((4, D + 1)))


def test_stability_high_with_signal_low_without() -> None:
    """The proposal's pilot check must separate 'real role signal' from
    'pure noise' (the dummy backend's bag mode)."""
    planted = _planted(2)
    agent, patient = _role_activations(planted, noise=0.3, seed=3)
    assert bootstrap_stability(agent, patient, n_resamples=100, seed=0) > 0.95

    rng = np.random.default_rng(4)
    noise_only = (rng.standard_normal((N, D)), rng.standard_normal((N, D)))
    assert bootstrap_stability(*noise_only, n_resamples=100, seed=0) < 0.6


def test_shuffled_label_direction_loses_role_signal() -> None:
    planted = _planted(5)
    agent, patient = _role_activations(planted, noise=0.3, seed=6)
    fitted = fit_role_direction(agent, patient)
    # A single shuffle retains planted overlap through chance label imbalance
    # (see the note in directions.fit), so the contract is statistical: across
    # shuffle seeds the mean |overlap| sits far below the real fit's.
    overlaps = []
    for seed in range(30):
        shuffled = shuffled_label_direction(agent, patient, seed=seed)
        assert np.linalg.norm(shuffled) == pytest.approx(1.0)
        overlaps.append(abs(float(shuffled @ planted)))
    assert float(fitted @ planted) > 0.97
    assert float(np.mean(overlaps)) < 0.7


def test_generic_loo_excludes_held_out_entity() -> None:
    directions = {"a": _planted(7), "b": _planted(8), "c": _planted(9)}
    loo = generic_loo_direction(directions, held_out="a")
    expected = directions["b"] + directions["c"]
    expected /= np.linalg.norm(expected)
    assert loo == pytest.approx(expected)
    with pytest.raises(ValueError, match="no entities besides"):
        generic_loo_direction({"a": directions["a"]}, held_out="a")


def test_fit_all_and_storage_roundtrip(tmp_path: Path) -> None:
    activations = {
        entity: _role_activations(_planted(seed), noise=0.3, seed=seed + 100)
        for seed, entity in enumerate(("doctor", "nurse", "chef"))
    }
    fitted = fit_all(activations, InjectionSite.FINAL_TOKEN, n_bootstrap=50, seed=0)
    assert fitted.entities == ("chef", "doctor", "nurse")  # sorted
    assert fitted.fitted.shape == (3, D)
    assert all(s > 0.9 for s in fitted.stability)

    path = save_directions(fitted, tmp_path)
    assert path.name == "directions_final_token.npz"
    loaded = load_directions(tmp_path, InjectionSite.FINAL_TOKEN)
    assert loaded.entities == fitted.entities
    assert loaded.fitted == pytest.approx(fitted.fitted)
    assert loaded.direction("doctor") == pytest.approx(fitted.direction("doctor"))
    with pytest.raises(KeyError, match="no direction fitted"):
        loaded.direction("pilot")
    with pytest.raises(FileNotFoundError, match="fit_directions"):
        load_directions(tmp_path, InjectionSite.ENTITY_TOKEN)
