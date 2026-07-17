"""Contract tests for analysis.stats (docs/ARCHITECTURE.md, Testing)."""

from __future__ import annotations

import numpy as np
import pytest

from jspace_binding.analysis.stats import (
    bootstrap_ci,
    cohens_d,
    cohens_d_ci,
    holm_bonferroni,
    null_band,
    permutation_pvalue,
)


def test_bootstrap_ci_covers_known_mean() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.1, 200)
    scores = list(0.5 + noise - noise.mean())  # sample mean exactly 0.5
    lo, hi = bootstrap_ci(scores, n_resamples=2000, ci_level=0.95, seed=0)
    assert lo < 0.5 < hi
    assert 0.0 < hi - lo < 0.06  # ~4 standard errors at n=200, sigma=0.1


def test_bootstrap_ci_deterministic_given_seed() -> None:
    scores = [0.1, 0.2, 0.4, 0.8]
    first = bootstrap_ci(scores, n_resamples=500, seed=7)
    assert bootstrap_ci(scores, n_resamples=500, seed=7) == first


def test_permutation_pvalue_small_for_shifted_sample() -> None:
    rng = np.random.default_rng(1)
    shifted = list(rng.normal(0.5, 0.05, 24))
    assert min(shifted) > 0.0  # clearly one-sided: every family agrees
    assert permutation_pvalue(shifted, n_permutations=2000, seed=0) < 0.01


def test_permutation_pvalue_large_for_zero_centered_sample() -> None:
    rng = np.random.default_rng(2)
    half = rng.normal(0.0, 1.0, 15)
    balanced = list(np.concatenate([half, -half]))  # mean exactly 0
    assert permutation_pvalue(balanced, n_permutations=2000, seed=0) > 0.5


def test_holm_bonferroni_step_down_stops_at_first_failure() -> None:
    pvalues = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5}
    # sorted: a (0.01 <= 0.05/4), then c (0.03 > 0.05/3) fails and blocks the rest.
    assert holm_bonferroni(pvalues, alpha=0.05) == {
        "a": True,
        "b": False,
        "c": False,
        "d": False,
    }


def test_holm_bonferroni_less_conservative_than_bonferroni() -> None:
    # Plain Bonferroni (alpha/m = 0.025) would retain y = 0.049; Holm rejects both.
    assert holm_bonferroni({"x": 0.012, "y": 0.049}, alpha=0.05) == {"x": True, "y": True}


def test_cohens_d_matches_hand_computation() -> None:
    # mean 3, sd (ddof=1) sqrt(2.5) -> d = 3 / sqrt(2.5)
    assert cohens_d([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(3.0 / np.sqrt(2.5))
    assert cohens_d([-2.0, -1.0, 1.0, 2.0]) == pytest.approx(0.0, abs=1e-12)


def test_cohens_d_ci_brackets_point_estimate_and_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    scores = list(rng.normal(0.8, 0.4, 60))
    lo, hi = cohens_d_ci(scores, n_resamples=2000, seed=0)
    assert lo < cohens_d(scores) < hi
    assert lo > 0.5  # clearly-shifted sample: the d CI excludes zero
    assert (lo, hi) == cohens_d_ci(scores, n_resamples=2000, seed=0)


def test_cohens_d_ci_tight_around_zero_for_null_sample() -> None:
    rng = np.random.default_rng(4)
    half = rng.normal(0.0, 0.3, 30)
    scores = list(np.concatenate([half, -half]))  # mean exactly 0
    lo, hi = cohens_d_ci(scores, n_resamples=2000, seed=0)
    assert lo < 0.0 < hi
    assert hi - lo < 1.0  # the "clean negative" tightness criterion


def test_null_band_is_percentile_band() -> None:
    scores = [float(x) for x in range(101)]
    lo, hi = null_band(scores, ci_level=0.95)
    assert lo == pytest.approx(2.5)
    assert hi == pytest.approx(97.5)
