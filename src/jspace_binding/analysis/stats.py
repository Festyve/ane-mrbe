"""Statistics over family-level binding scores. Pure numpy.

The unit of resampling and permutation is the ItemFamily, which contributes
one BS_i (see analysis.binding_score). The pairing lives inside BS_i, so
family-level resampling preserves it, and a within-family agent/patient label
swap negates BS_i exactly, keeping the sign-flip permutation test exact. All
randomness flows through a local np.random.default_rng(seed), so every number
is reproducible from the config seed alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

K = TypeVar("K")


def bootstrap_ci(
    scores: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean binding score.

    Resamples families with replacement (families, not trials — the DiD
    pairing must travel with each BS_i), takes the mean of each resample, and
    returns the ((1-ci_level)/2, 1-(1-ci_level)/2) percentiles of the
    n_resamples means.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci: need at least one family-level score")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    tail = 100.0 * (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(means, [tail, 100.0 - tail])
    return float(lo), float(hi)


def permutation_pvalue(
    scores: Sequence[float] | np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided sign-flip permutation test of mean(BS_i) = 0.

    Under H0 the agent/patient labels within a family are exchangeable, and
    swapping them is exactly a sign flip of that family's score. The null is
    built by drawing s_i in {-1, +1} uniformly and recomputing
    mean(s_i * BS_i). Two-sided, with the add-one correction that keeps p > 0:
    p = (1 + #{|T_perm| >= |T_obs|}) / (1 + n_permutations).
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("permutation_pvalue: need at least one family-level score")
    rng = np.random.default_rng(seed)
    observed = abs(arr.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, arr.size))
    permuted = np.abs((signs * arr).mean(axis=1))
    return float((1 + np.count_nonzero(permuted >= observed)) / (1 + n_permutations))


def cohens_d(scores: Sequence[float] | np.ndarray) -> float:
    """One-sample Cohen's d against 0: d = mean(BS_i) / sd(BS_i), ddof=1.

    Raises rather than guessing on degenerate inputs: fewer than 2 scores
    leave the ddof=1 sd undefined, and sd == 0 makes d infinite — both would
    silently poison downstream summaries.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size < 2:
        raise ValueError("cohens_d: need >= 2 scores for a ddof=1 standard deviation")
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        raise ValueError("cohens_d: zero variance across scores — d is undefined")
    return float(arr.mean() / sd)


def cohens_d_ci(
    scores: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI on Cohen's d itself, resampling families.

    Degenerate resamples (zero variance, which would make d infinite) are
    dropped; if every resample is degenerate the input is unusable and we
    raise. The CI width separates "clean negative" from "underpowered" in the
    outcome classification.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size < 2:
        raise ValueError("cohens_d_ci: need >= 2 scores")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    samples = arr[idx]
    sds = samples.std(axis=1, ddof=1)
    valid = sds > 0.0
    if not np.any(valid):
        raise ValueError("cohens_d_ci: all bootstrap resamples degenerate (zero variance)")
    ds = samples.mean(axis=1)[valid] / sds[valid]
    tail = 100.0 * (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(ds, [tail, 100.0 - tail])
    return float(lo), float(hi)


def holm_bonferroni(pvalues: dict[K, float], alpha: float = 0.05) -> dict[K, bool]:
    """Holm-Bonferroni step-down over the per-group p-values.

    Sort the m p-values ascending; reject the k-th smallest while
    p_(k) <= alpha / (m - k). Controls family-wise error at alpha with no
    independence assumption, over one mean BS per (construction x pair) group.
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    rejected: dict[K, bool] = {}
    rejecting = True
    for k, (key, p) in enumerate(ordered):
        rejecting = rejecting and p <= alpha / (m - k)
        rejected[key] = rejecting
    return rejected


def null_band(
    scores: Sequence[float] | np.ndarray,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Percentile band of control-edit binding scores.

    Controls run the same DiD as the real edit but carry no role information,
    so their scores sample the no-binding distribution directly. The returned
    percentiles are the band the real-edit mean must clear.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("null_band: no control scores provided")
    tail = 100.0 * (1.0 - ci_level) / 2.0
    lo, hi = np.percentile(arr, [tail, 100.0 - tail])
    return float(lo), float(hi)
