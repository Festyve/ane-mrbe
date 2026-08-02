"""E4 (on-demand recruitment) against DummyModel ground truth.

The experiment's whole claim is that it separates three hypotheses the other
experiments conflate, so the tests check exactly that: each dummy mode must
produce its own verdict, and `binding` vs `recruitment` — indistinguishable to
RQ1/RQ2/primary — must come apart here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jspace_binding.config import AnalysisConfig, Config, PathsConfig, StimuliConfig
from jspace_binding.experiments.recruitment import BAG_CONDITION, ROLE_CONDITION, run_e4
from jspace_binding.experiments.rq1_probe import run_rq1
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.generate import generate_families


def _config(tmp_path: Path) -> Config:
    return Config(
        stimuli=StimuliConfig(items_per_cell=4),
        analysis=AnalysisConfig(n_bootstrap=100, n_permutation=100),
        paths=PathsConfig(
            stimuli=tmp_path / "stimuli.jsonl",
            fitting_corpus=tmp_path / "fitting_corpus.jsonl",
            directions=tmp_path / "directions",
            calibration=tmp_path / "calibration.json",
            results=tmp_path / "results",
            figures=tmp_path / "figures",
        ),
    )


def _e4(tmp_path: Path, mode: str) -> dict:
    config = _config(tmp_path)
    summary = run_e4(config, DummyModel(mode=mode, seed=0), generate_families(config))
    json.dumps(summary)  # must be serializable as returned
    assert Path(summary["figure"]).stat().st_size > 0
    assert (Path(config.paths.results) / "e4_recruitment.json").exists()
    return summary


def test_e4_recruitment_mode_shows_on_demand_binding(tmp_path: Path) -> None:
    """H3: role decodable from jspace under the role question, at chance under
    the bag question."""
    jspace = _e4(tmp_path, "recruitment")["sources"]["jspace"]
    assert jspace[ROLE_CONDITION]["accuracy"] > 0.9
    assert abs(jspace[BAG_CONDITION]["accuracy"] - 0.5) < 0.1
    assert jspace["recruitment_delta"] > 0.3
    assert jspace["verdict"] == "recruited"


def test_e4_binding_mode_shows_always_on_binding(tmp_path: Path) -> None:
    """An always-on workspace holds role under BOTH questions, so the delta is
    ~0 — a different verdict from H3 despite both being 'binding' models."""
    jspace = _e4(tmp_path, "binding")["sources"]["jspace"]
    assert jspace[ROLE_CONDITION]["accuracy"] > 0.9
    assert jspace[BAG_CONDITION]["accuracy"] > 0.9
    assert abs(jspace["recruitment_delta"]) < 0.15
    assert jspace["verdict"] == "always_on"


def test_e4_bag_mode_shows_no_role_in_jspace(tmp_path: Path) -> None:
    """A bag workspace never carries role, so neither question decodes it."""
    jspace = _e4(tmp_path, "bag")["sources"]["jspace"]
    assert abs(jspace[ROLE_CONDITION]["accuracy"] - 0.5) < 0.1
    assert abs(jspace[BAG_CONDITION]["accuracy"] - 0.5) < 0.1
    assert jspace["verdict"] == "absent"


def test_e4_separates_what_rq1_cannot(tmp_path: Path) -> None:
    """The reason E4 exists, asserted rather than assumed.

    RQ1 reads with no question in context, so an always-on workspace and an
    on-demand one are identical to it. If this ever stopped being true, E4
    would be redundant and RQ1 would be measuring something it does not claim
    to.
    """
    rq1_binding = run_rq1(
        _config(tmp_path / "a"), DummyModel(mode="binding", seed=0),
        generate_families(_config(tmp_path / "a")),
    )
    rq1_recruit = run_rq1(
        _config(tmp_path / "b"), DummyModel(mode="recruitment", seed=0),
        generate_families(_config(tmp_path / "b")),
    )
    for site in rq1_binding["sites"]:
        assert (
            rq1_binding["sites"][site]["jspace"]["accuracy"]
            == rq1_recruit["sites"][site]["jspace"]["accuracy"]
        ), "RQ1 distinguished binding from recruitment; it cannot, and must not"

    # E4, on the same two models, must tell them apart.
    binding = _e4(tmp_path / "c", "binding")["verdict"]
    recruit = _e4(tmp_path / "d", "recruitment")["verdict"]
    assert binding != recruit
    assert {binding, recruit} == {"always_on", "recruited"}


def test_e4_recruitment_exceeds_the_capacity_control(tmp_path: Path) -> None:
    """A jspace delta means nothing if a same-rank random subspace shows the
    same one — that would be the question shifting the residual generally, not
    recruitment INTO the workspace."""
    summary = _e4(tmp_path, "recruitment")
    assert summary["exceeds_capacity_control"] is True
    jspace = summary["sources"]["jspace"]["recruitment_delta"]
    random_subspace = summary["sources"]["random_subspace"]["recruitment_delta"]
    assert jspace > random_subspace


def test_e4_reports_per_fold_accuracies(tmp_path: Path) -> None:
    """With one fold per concept pair the mean can hide an inverted pair, so
    the folds travel with it — same guardrail as RQ1."""
    summary = _e4(tmp_path, "recruitment")
    for by_condition in summary["sources"].values():
        for condition in (ROLE_CONDITION, BAG_CONDITION):
            block = by_condition[condition]
            assert len(block["fold_accuracies"]) == block["n_folds"]
            assert len(block["fold_ids"]) == block["n_folds"]
            mean = sum(block["fold_accuracies"]) / block["n_folds"]
            assert abs(mean - block["accuracy"]) < 1e-9


def test_e4_requires_both_questions(tmp_path: Path) -> None:
    """A family missing either probe cannot be compared across conditions, and
    scoring it would silently drop one arm of the contrast."""
    import dataclasses

    config = _config(tmp_path)
    families = generate_families(config)
    stripped = [dataclasses.replace(f, neutral_probe="") for f in families]
    with pytest.raises(ValueError, match="role or neutral probe"):
        run_e4(config, DummyModel(mode="recruitment", seed=0), stripped)


def test_e4_below_chance_is_not_reported_as_presence() -> None:
    """Two below-chance conditions must never read as `always_on`.

    The real Qwen3.6-27B run came back role 0.281 / bag 0.263 -- both far below
    chance, every fold inverting -- and the first version of _verdict reported
    `always_on`, i.e. "binding is continuously present". A probe scoring 0.26
    under leave-one-pair-out has learned a rule that anti-transfers across
    concept pairs; that is the opposite of evidence for presence, and it would
    have gone into a writeup as the headline verdict.
    """
    from jspace_binding.experiments.recruitment import _verdict

    assert _verdict(0.281, 0.263, 0.018) == "anti_transfer_both"
    assert _verdict(0.28, 0.55, -0.27) == "anti_transfer_one"
    # Genuine presence still reads as presence.
    assert _verdict(0.95, 0.93, 0.02) == "always_on"
    assert _verdict(0.95, 0.50, 0.45) == "recruited"
    assert _verdict(0.50, 0.50, 0.00) == "absent"
