"""Hand-computed DiD cases for the binding score (docs/ARCHITECTURE.md, Testing)."""

from __future__ import annotations

import pytest

from jspace_binding.analysis.binding_score import (
    collect_scores,
    control_binding_score,
    family_binding_score,
    position_average,
)
from jspace_binding.types import (
    Construction,
    EditType,
    InjectionSite,
    Position,
    ProbeKind,
    Role,
    TrialResult,
)

REAL, NO_EDIT = EditType.REAL, EditType.NO_EDIT
NULL, RANDOM = EditType.NULL_NON_PARTICIPANT, EditType.RANDOM_DIRECTION

# Worked-example P(target) levels at the ROLE probe: {edit: (agent, patient)}.
# binding: mass moves only when the swap hits the probed role -> BS = 0.55;
# bag: the swap is role-blind -> BS = 0.
BINDING = {REAL: (0.60, 0.05), NO_EDIT: (0.01, 0.01)}
BAG = {REAL: (0.30, 0.30), NO_EDIT: (0.01, 0.01)}


def _trial(
    role: Role,
    position: Position,
    edit_type: EditType,
    p_target: float,
    *,
    probe: ProbeKind = ProbeKind.ROLE,
    site: InjectionSite = InjectionSite.FINAL_TOKEN,
    family_id: str = "fam-0",
    pair_id: str = "doctor->nurse",
    construction: Construction = Construction.ACTIVE_PASSIVE,
    tokens: tuple[str, str, str] = ("doctor", "nurse", "lawyer"),
) -> TrialResult:
    source, target, other = tokens
    return TrialResult(
        family_id=family_id,
        pair_id=pair_id,
        construction=construction,
        role=role,
        position=position,
        edit_type=edit_type,
        probe_kind=probe,
        injection_site=site,
        answer_probs={target: p_target, source: 0.02, other: 0.02},
    )


def _family_trials(
    p_by_edit: dict[EditType, tuple[float, float]], bias: float = 0.0, **kwargs: object
) -> list[TrialResult]:
    """One ROLE-probe trial per (role, position, edit); optional FIRST-position bias."""
    trials = []
    for edit_type, (p_agent, p_patient) in p_by_edit.items():
        for role, p in ((Role.AGENT, p_agent), (Role.PATIENT, p_patient)):
            for position in Position:
                p_cell = p + (bias if position is Position.FIRST else 0.0)
                trials.append(_trial(role, position, edit_type, p_cell, **kwargs))
    return trials


def test_position_average_collapses_positions() -> None:
    probs = {
        (Role.AGENT, Position.FIRST): 0.6,
        (Role.AGENT, Position.SECOND): 0.5,
        (Role.PATIENT, Position.FIRST): 0.2,
        (Role.PATIENT, Position.SECOND): 0.1,
    }
    assert position_average(probs) == pytest.approx({Role.AGENT: 0.55, Role.PATIENT: 0.15})


def test_worked_example_binding_score() -> None:
    assert family_binding_score(_family_trials(BINDING), "nurse") == pytest.approx(0.55)


def test_worked_example_bag_score() -> None:
    assert family_binding_score(_family_trials(BAG), "nurse") == pytest.approx(0.0, abs=1e-12)


def test_position_bias_cancels_under_averaging() -> None:
    for levels in (BINDING, BAG):
        unbiased = family_binding_score(_family_trials(levels), "nurse")
        biased = family_binding_score(_family_trials(levels, bias=0.03), "nurse")
        assert biased == pytest.approx(unbiased)


def test_neutral_probe_trials_are_ignored() -> None:
    trials = _family_trials(BINDING) + [
        _trial(role, position, REAL, 0.99, probe=ProbeKind.NEUTRAL)
        for role in Role
        for position in Position
    ]
    assert family_binding_score(trials, "nurse") == pytest.approx(0.55)


def test_control_binding_score_runs_the_same_did() -> None:
    trials = _family_trials({NULL: (0.21, 0.11), RANDOM: (0.01, 0.01), NO_EDIT: (0.01, 0.01)})
    assert control_binding_score(trials, "nurse", NULL) == pytest.approx(0.10)
    assert control_binding_score(trials, "nurse", RANDOM) == pytest.approx(0.0, abs=1e-12)


def test_collect_scores_groups_by_construction_and_pair() -> None:
    fam_a = _family_trials({**BINDING, NULL: (0.01, 0.01), RANDOM: (0.01, 0.01)})
    fam_b = _family_trials(
        {**BAG, NULL: (0.01, 0.01), RANDOM: (0.01, 0.01)},
        family_id="fam-1",
        pair_id="teacher->student",
        construction=Construction.DATIVE,
        tokens=("teacher", "student", "judge"),
    )
    # The same family re-measured at the entity-token site must not contaminate
    # (or, via the duplicate-cell guard, break) the final-token table.
    entity_site = [
        _trial(role, position, REAL, 0.99, site=InjectionSite.ENTITY_TOKEN)
        for role in Role
        for position in Position
    ]
    table = collect_scores(fam_a + fam_b + entity_site, InjectionSite.FINAL_TOKEN)
    assert table.real == {
        ("active_passive", "doctor->nurse"): [pytest.approx(0.55)],
        ("dative", "teacher->student"): [pytest.approx(0.0, abs=1e-12)],
    }
    assert table.null_band == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)
