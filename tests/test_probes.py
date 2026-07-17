"""Contract tests for analysis.probes (RQ1)."""

from __future__ import annotations

import numpy as np
import pytest

from jspace_binding.analysis.probes import (
    ProbeExample,
    fit_ridge_probe,
    leave_one_pair_out,
    probe_accuracy,
)

D = 12


def _examples(signal: bool, n_per_pair: int = 30, seed: int = 0) -> list[ProbeExample]:
    """Synthetic examples over three pairs; role signal along a planted
    direction when signal=True, pure noise otherwise."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(D)
    direction /= np.linalg.norm(direction)
    out = []
    for pair_id in ("doctor->nurse", "teacher->student", "driver->passenger"):
        for i in range(n_per_pair):
            is_agent = i % 2 == 0
            base = rng.standard_normal(D) * 0.5
            if signal:
                base = base + (1.0 if is_agent else -1.0) * direction
            out.append(
                ProbeExample(
                    pair_id=pair_id, is_agent=is_agent, features=tuple(float(x) for x in base)
                )
            )
    return out


def test_ridge_probe_separates_separable_data() -> None:
    examples = _examples(signal=True)
    X = np.asarray([ex.features for ex in examples])
    y = np.asarray([1.0 if ex.is_agent else -1.0 for ex in examples])
    w = fit_ridge_probe(X, y)
    assert probe_accuracy(w, X, y) > 0.95


def test_leave_one_pair_out_decodes_signal_generalizably() -> None:
    report = leave_one_pair_out(_examples(signal=True))
    assert report.n_folds == 3
    assert report.accuracy > 0.9
    # Control task (arbitrary labels) must sit near chance for a low-capacity
    # linear probe: high selectivity = genuine role information.
    assert abs(report.control_accuracy - 0.5) < 0.15
    assert report.selectivity > 0.35


def test_leave_one_pair_out_at_chance_without_signal() -> None:
    report = leave_one_pair_out(_examples(signal=False))
    assert abs(report.accuracy - 0.5) < 0.15
    assert abs(report.selectivity) < 0.2


def test_split_requires_multiple_pairs() -> None:
    single = [ex for ex in _examples(signal=True) if ex.pair_id == "doctor->nurse"]
    with pytest.raises(ValueError, match="need >= 2 concept pairs"):
        leave_one_pair_out(single)
