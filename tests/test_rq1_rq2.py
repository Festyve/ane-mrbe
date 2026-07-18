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


def _rq2(tmp_path: Path, mode: str) -> dict:
    config = _config(tmp_path)
    summary = run_rq2(config, DummyModel(mode=mode, seed=0), generate_families(config))
    json.dumps(summary)
    assert Path(summary["figure"]).stat().st_size > 0
    assert summary["difficulty_matched"] is True  # same sentences, same baselines
    return summary


def test_rq2_binding_mode_shows_binding_specific_deficit(tmp_path: Path) -> None:
    summary = _rq2(tmp_path, "binding")
    jspace = summary["deltas"][EditType.ABLATE_JSPACE.value]
    random_sub = summary["deltas"][EditType.ABLATE_RANDOM_SUBSPACE.value]
    assert jspace["binding_specific_deficit"] > 0.3
    assert abs(random_sub["binding_specific_deficit"]) < 0.15
    assert summary["workspace_causally_involved"] is True


def test_rq2_bag_mode_shows_no_binding_specific_deficit(tmp_path: Path) -> None:
    summary = _rq2(tmp_path, "bag")
    jspace = summary["deltas"][EditType.ABLATE_JSPACE.value]
    # Recall collapses at least as hard as binding: no binding-SPECIFIC deficit.
    assert jspace["binding_specific_deficit"] < 0.15
    assert summary["workspace_causally_involved"] is False
