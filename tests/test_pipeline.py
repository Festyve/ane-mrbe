"""End-to-end dry runs against DummyModel ground truth (docs/ARCHITECTURE.md, Testing).

binding mode must recover a large pooled binding score that clears the control
null band; bag mode must land near 0 inside it. This validates the analysis
code against known answers before the real model is ever loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from jspace_binding.config import AnalysisConfig, Config, PathsConfig, StimuliConfig
from jspace_binding.experiments.primary import analyze, run_primary
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.templates import VERBS_BY_CONSTRUCTION, build_family
from jspace_binding.stimuli.vocab import OTHER_ENTITY_BY_PAIR
from jspace_binding.types import Construction, ItemFamily


def _tiny_config(tmp_path: Path) -> Config:
    return Config(
        stimuli=StimuliConfig(items_per_cell=2),
        analysis=AnalysisConfig(n_bootstrap=500, n_permutation=500),
        paths=PathsConfig(
            stimuli=tmp_path / "stimuli.jsonl",
            results=tmp_path / "results",
            figures=tmp_path / "figures",
        ),
    )


def _families(config: Config) -> list[ItemFamily]:
    """All four constructions, at the tiny config's scale."""
    return [
        build_family(
            pair=pair,
            construction=construction,
            other_entity=OTHER_ENTITY_BY_PAIR.get(pair.pair_id, "lawyer"),
            verb_lemma=VERBS_BY_CONSTRUCTION[construction][
                index % len(VERBS_BY_CONSTRUCTION[construction])
            ],
            family_index=index,
        )
        for construction in Construction
        for pair in config.stimuli.concept_pairs
        for index in range(config.stimuli.items_per_cell)
    ]


def _run(tmp_path: Path, mode: str) -> dict:
    config = _tiny_config(tmp_path)
    families = _families(config)
    assert families, "the pinned active_passive construction must always build"
    trials = run_primary(config, DummyModel(mode=mode, seed=0), families)
    # ROLE probe for every edit x site, NEUTRAL additionally for REAL:
    # per cell that is n_edits * n_sites + n_sites trials.
    n_edits = len(config.experiment.edit_types)
    n_sites = len(config.experiment.injection_sites)
    assert len(trials) == len(families) * 4 * (n_edits * n_sites + n_sites)
    assert (Path(config.paths.results) / "trials.jsonl").exists()
    summary = analyze(config, trials)
    json.dumps(summary)  # must be JSON-serializable exactly as returned
    for figure in summary["figures"].values():
        assert Path(figure).stat().st_size > 0
    return summary


def test_binding_mode_recovers_large_pooled_score(tmp_path: Path) -> None:
    summary = _run(tmp_path, "binding")
    band_hi = summary["null_band"][1]
    assert summary["pooled"]["mean"] > 0.3
    assert summary["pooled"]["mean"] > band_hi


def test_bag_mode_lands_inside_null_band(tmp_path: Path) -> None:
    summary = _run(tmp_path, "bag")
    band_lo, band_hi = summary["null_band"]
    mean = summary["pooled"]["mean"]
    assert abs(mean) < 0.1
    assert band_lo <= mean <= band_hi
