"""End-to-end dry runs against DummyModel ground truth (docs/ARCHITECTURE.md, Testing).

binding mode must recover a large pooled binding score (the ~2.2-logit
crossover encoded in the dummy) that clears the control null band, with both
per-sign gap changes negative; bag mode must land near 0 inside the band.
Both modes must pass the neutral-probe strength check (the identity swap
"works" role-independently in either world). This validates the analysis code
against known answers before the real model is ever loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from jspace_binding.config import AnalysisConfig, Config, PathsConfig, StimuliConfig
from jspace_binding.experiments.primary import analyze, run_primary
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.templates import VERBS_BY_CONSTRUCTION, build_family
from jspace_binding.stimuli.vocab import OTHER_ENTITY_BY_PAIR
from jspace_binding.types import (
    DIRECTION_PUSH_EDIT_TYPES,
    Construction,
    ItemFamily,
    PushSign,
)


def _tiny_config(tmp_path: Path) -> Config:
    return Config(
        stimuli=StimuliConfig(items_per_cell=2),
        analysis=AnalysisConfig(n_bootstrap=500, n_permutation=500),
        paths=PathsConfig(
            stimuli=tmp_path / "stimuli.jsonl",
            fitting_corpus=tmp_path / "fitting_corpus.jsonl",
            directions=tmp_path / "directions",
            calibration=tmp_path / "calibration.json",
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
    # Per cell x site: each direction-push edit runs both signs at the ROLE
    # probe, NO_EDIT runs ROLE + NEUTRAL, IDENTITY_SWAP runs NEUTRAL only.
    n_push = sum(1 for e in config.experiment.edit_types if e in DIRECTION_PUSH_EDIT_TYPES)
    per_cell_site = n_push * len(config.experiment.push_signs) + 2 + 1
    n_sites = len(config.experiment.injection_sites)
    assert len(trials) == len(families) * 4 * n_sites * per_cell_site
    assert (Path(config.paths.results) / "trials.jsonl").exists()
    summary = analyze(config, trials)
    json.dumps(summary)  # must be JSON-serializable exactly as returned
    for figure in summary["figures"].values():
        assert Path(figure).stat().st_size > 0
    return summary


def test_binding_mode_recovers_large_pooled_score(tmp_path: Path) -> None:
    summary = _run(tmp_path, "binding")
    band_hi = summary["null_band"][1]
    assert summary["pooled"]["mean"] > 1.0  # dummy plants ~2.2 logits
    assert summary["pooled"]["mean"] > band_hi
    # The crossover signature: the gap shrinks from opposite sides, so BOTH
    # signed gap changes are negative.
    for sign in PushSign:
        assert summary["gap_change_by_sign"][sign.value]["mean"] < -1.0
    assert summary["neutral_strength_check"]["passes"] is True
    # Verdict: with the strength check passing, a pooled-significant d >= 0.5
    # effect classifies as at least suggestive; positive_binding additionally
    # needs per-construction Holm significance, unreachable at tiny-CI n.
    verdict = summary["verdict"]
    assert verdict["strength_check_passes"] is True
    assert verdict["pooled_significant"] is True
    assert verdict["outcome"] in ("positive_binding", "suggestive_not_conclusive")
    assert verdict["random_direction_in_band"] is True
    # d carries its own bootstrap CI, excluding zero here.
    assert summary["pooled"]["cohens_d_ci_lo"] > 0.0
    # Per-construction blocks exist for all four constructions.
    assert set(summary["constructions"]) == {
        "active_passive", "dative", "cleft", "relative_clause"
    }


def test_bag_mode_lands_inside_null_band(tmp_path: Path) -> None:
    summary = _run(tmp_path, "bag")
    band_lo, band_hi = summary["null_band"]
    mean = summary["pooled"]["mean"]
    assert abs(mean) < 0.15
    assert band_lo <= mean <= band_hi
    # A uniform log-odds push leaves each signed gap unchanged up to jitter.
    for sign in PushSign:
        assert abs(summary["gap_change_by_sign"][sign.value]["mean"]) < 0.15
    # The strength check still passes: "the edit works" is exactly what makes
    # a bag-mode null interpretable as a clean negative.
    assert summary["neutral_strength_check"]["passes"] is True
    # And the verdict says so: not significant, tight CI on d -> clean negative,
    # reported as a positive finding for the falsifiable claim.
    assert summary["verdict"]["outcome"] == "clean_negative"
