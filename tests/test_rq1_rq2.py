"""End-to-end RQ1/RQ2 dry runs against DummyModel ground truth.

RQ1: binding mode plants the role signal in the J-space component (jspace +
residual decode, orthogonal at chance); bag mode plants it in the orthogonal
remainder (jspace at chance). RQ2: binding mode shows a binding-specific
J-space ablation deficit exceeding the random-subspace bar; bag mode shows
none (recall collapses at least as hard as binding).
"""

from __future__ import annotations

import json
from pathlib import Path

from jspace_binding.config import AnalysisConfig, Config, PathsConfig, StimuliConfig
from jspace_binding.experiments.rq1_probe import run_rq1
from jspace_binding.experiments.rq2_ablation import run_rq2
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.types import EditType


def _config(tmp_path: Path) -> Config:
    return Config(
        stimuli=StimuliConfig(items_per_cell=4),
        analysis=AnalysisConfig(n_bootstrap=200, n_permutation=200),
        paths=PathsConfig(
            stimuli=tmp_path / "stimuli.jsonl",
            fitting_corpus=tmp_path / "fitting_corpus.jsonl",
            directions=tmp_path / "directions",
            calibration=tmp_path / "calibration.json",
            results=tmp_path / "results",
            figures=tmp_path / "figures",
        ),
    )


def _rq1(tmp_path: Path, mode: str) -> dict:
    config = _config(tmp_path)
    summary = run_rq1(config, DummyModel(mode=mode, seed=0), generate_families(config))
    json.dumps(summary)
    assert Path(summary["figure"]).stat().st_size > 0
    assert (Path(config.paths.results) / "rq1_probe.json").exists()
    return summary


def test_rq1_binding_mode_localizes_role_in_jspace(tmp_path: Path) -> None:
    sites = _rq1(tmp_path, "binding")["sites"]
    for by_source in sites.values():
        assert by_source["jspace"]["accuracy"] > 0.9
        assert by_source["residual"]["accuracy"] > 0.9
        assert abs(by_source["orthogonal"]["accuracy"] - 0.5) < 0.2
        assert by_source["jspace"]["selectivity"] > 0.3


def test_rq1_bag_mode_localizes_role_outside_jspace(tmp_path: Path) -> None:
    sites = _rq1(tmp_path, "bag")["sites"]
    for by_source in sites.values():
        assert abs(by_source["jspace"]["accuracy"] - 0.5) < 0.2
        assert by_source["orthogonal"]["accuracy"] > 0.9
        assert by_source["residual"]["accuracy"] > 0.9


def test_rq1_reports_per_fold_accuracies(tmp_path: Path) -> None:
    """The mean alone cannot distinguish weak transfer from an inverted fold."""
    summary = _rq1(tmp_path, "binding")
    for by_source in summary["sites"].values():
        for report in by_source.values():
            assert len(report["fold_accuracies"]) == report["n_folds"]
            assert len(report["fold_ids"]) == report["n_folds"]
            assert len(report["control_fold_accuracies"]) == report["n_folds"]
            # The mean must be reconstructible from the folds it summarizes.
            mean = sum(report["fold_accuracies"]) / report["n_folds"]
            assert abs(mean - report["accuracy"]) < 1e-9
    # Dummy ground truth transfers across pairs, so nothing should invert.
    assert summary["inverting_folds_present"] is False


def test_rq1_flags_a_fold_that_inverts() -> None:
    """A pair the probe scores below chance is named, not averaged away."""
    from jspace_binding.analysis.probes import ProbeExample, leave_one_pair_out

    rows = []
    for pair in ["a", "b", "c", "d"]:
        # Pair "d" encodes role with the opposite sign from the other three, so
        # the rule learned on a+b+c runs exactly backwards when d is held out.
        sign = -1.0 if pair == "d" else 1.0
        for i in range(40):
            is_agent = i % 2 == 0
            rows.append(
                ProbeExample(
                    pair_id=pair,
                    is_agent=is_agent,
                    features=(sign * (1.0 if is_agent else -1.0),),
                )
            )
    report = leave_one_pair_out(rows)
    assert report.inverting_folds == ("d",)
    assert report.fold_spread == 1.0
    # The headline mean stays high and healthy-looking (three folds at 1.0)
    # while one pair decodes exactly backwards. This is precisely what the
    # mean conceals and the per-fold list exposes.
    assert report.accuracy == 0.75
    assert dict(zip(report.fold_ids, report.fold_accuracies, strict=True))["d"] == 0.0


def _rq2(tmp_path: Path, mode: str) -> dict:
    config = _config(tmp_path)
    summary = run_rq2(config, DummyModel(mode=mode, seed=0), generate_families(config))
    json.dumps(summary)
    # Every configured site is scored, not just a hard-coded one.
    assert set(summary["sites"]) == {s.value for s in config.experiment.injection_sites}
    for site in summary["sites"].values():
        assert Path(site["figure"]).stat().st_size > 0
        assert site["difficulty_matched"] is True  # same sentences, same baselines
    return summary


def test_rq2_binding_mode_shows_binding_specific_deficit(tmp_path: Path) -> None:
    summary = _rq2(tmp_path, "binding")
    for site in summary["sites"].values():
        jspace = site["deltas"][EditType.ABLATE_JSPACE.value]
        random_sub = site["deltas"][EditType.ABLATE_RANDOM_SUBSPACE.value]
        assert jspace["binding_specific_deficit"] > 0.3
        assert abs(random_sub["binding_specific_deficit"]) < 0.15
        # A real deficit must also clear zero by the family-level bootstrap.
        assert jspace["ci_excludes_zero"] is True
    assert summary["workspace_causally_involved"] is True


def test_rq2_bag_mode_shows_no_binding_specific_deficit(tmp_path: Path) -> None:
    summary = _rq2(tmp_path, "bag")
    for site in summary["sites"].values():
        deficit = site["deltas"][EditType.ABLATE_JSPACE.value]["binding_specific_deficit"]
        # Recall collapses at least as hard as binding: no binding-SPECIFIC deficit.
        assert deficit < 0.15
    assert summary["workspace_causally_involved"] is False


def test_rq2_records_provenance_and_control_health(tmp_path: Path) -> None:
    """The summary must say what was run, and whether its null is readable."""
    summary = _rq2(tmp_path, "binding")
    provenance = summary["provenance"]
    assert provenance["backend"] == "dummy"
    assert provenance["n_concept_pairs"] == 3
    assert provenance["n_families"] > 0
    for key in ("model_id", "layer_band", "jspace_k", "ablate_k", "git_commit"):
        assert key in provenance
    for site in summary["sites"].values():
        assert len(site["deltas"][EditType.ABLATE_JSPACE.value]["binding_specific_ci"]) == 2
        # KNOWN LIMITATION, asserted so it cannot regress silently: the recall
        # task sits at 1.0 under every condition, here and on the real model.
        # It therefore cannot register damage, the binding-minus-recall
        # subtraction reduces to the raw binding deficit, and no run using this
        # neutral probe can support a "binding-SPECIFIC" claim. Flip this to
        # True once the recall probe is redesigned off ceiling.
        assert site["accuracy"]["no_edit|neutral"] == 1.0
        assert site["recall_control_informative"] is False
