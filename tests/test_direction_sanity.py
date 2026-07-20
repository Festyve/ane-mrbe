"""Tests for analysis.direction_sanity: held-out separation + direction compare."""

from __future__ import annotations

import numpy as np
import pytest

from jspace_binding.analysis.direction_sanity import (
    compare_directions,
    cosine_matrix,
    project,
    scatter_projections,
    separation_report,
)

D = 16


def _planted(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(D)
    return raw / np.linalg.norm(raw)


def _held_out(planted: np.ndarray, noise: float, n: int, seed: int):
    rng = np.random.default_rng(seed)
    agent = planted + noise * rng.standard_normal((n, D))
    patient = -planted + noise * rng.standard_normal((n, D))
    return agent, patient


def test_project_dim_mismatch() -> None:
    with pytest.raises(ValueError, match="direction dim"):
        project(np.ones((3, D)), np.ones(D + 1))


def test_separation_clean_signal_is_near_perfect() -> None:
    planted = _planted(0)
    agent, patient = _held_out(planted, noise=0.2, n=50, seed=1)
    res = separation_report(project(agent, planted), project(patient, planted), "doctor", "final")
    assert res.auc > 0.95
    assert res.accuracy > 0.9
    assert res.cohens_d > 2.0
    assert res.separates


def test_separation_pure_noise_is_chance() -> None:
    """A direction unrelated to the labels must land at ~chance AUC — the
    control that proves the metric is not inflating separation."""
    rng = np.random.default_rng(2)
    agent = rng.standard_normal((60, D))
    patient = rng.standard_normal((60, D))
    unrelated = _planted(99)
    res = separation_report(project(agent, unrelated), project(patient, unrelated), "x", "s")
    assert 0.35 < res.auc < 0.65
    assert not res.separates


def test_separation_is_sign_agnostic() -> None:
    """A flipped direction (patient projects higher) must not read as chance:
    orientation is inferred from the data."""
    planted = _planted(3)
    agent, patient = _held_out(planted, noise=0.2, n=40, seed=4)
    flipped = -planted
    res = separation_report(project(agent, flipped), project(patient, flipped), "e", "s")
    assert res.auc < 0.05  # agent now projects LOW -> raw ranking near 0
    assert res.accuracy > 0.9  # ...but the classifier orients itself


def test_separation_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty projection set"):
        separation_report(np.array([]), np.ones(3), "e", "s")


def test_cosine_matrix_and_compare_verdicts() -> None:
    # Identical directions -> near-1 mean cosine -> filler-general verdict.
    d = _planted(5)
    same = compare_directions({"a": d, "b": d.copy()})
    assert same["mean_off_diagonal_cosine"] == pytest.approx(1.0)
    assert "filler-general" in same["verdict"]

    # Orthogonal-ish random directions -> low cosine -> distinct verdict.
    distinct = compare_directions({"a": _planted(6), "b": _planted(7), "c": _planted(8)})
    assert distinct["mean_off_diagonal_cosine"] < 0.5
    assert "distinct" in distinct["verdict"]

    # Shared component + noise -> moderate cosine -> related-but-distinct.
    base = _planted(9)
    shared = {name: base + 0.8 * _planted(10 + i) for i, name in enumerate("abc")}
    rel = compare_directions(shared)
    assert 0.3 <= rel["mean_off_diagonal_cosine"] < 0.9
    assert "related but distinct" in rel["verdict"]


def test_cosine_matrix_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        cosine_matrix({"a": _planted(1), "b": np.zeros(D)})


def test_compare_needs_two() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        compare_directions({"solo": _planted(1)})


def test_scatter_returns_path_or_none(tmp_path) -> None:
    """Never raises on a missing/broken matplotlib; returns None or a path."""
    planted = _planted(0)
    agent, patient = _held_out(planted, noise=0.2, n=10, seed=1)
    out = scatter_projections(
        {"doctor@final_token": (project(agent, planted), project(patient, planted))},
        tmp_path / "scatter.png",
    )
    assert out is None or out.exists()
