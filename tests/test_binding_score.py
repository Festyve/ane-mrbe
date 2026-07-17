"""Hand-computed crossover cases for the log-odds binding score."""

from __future__ import annotations

import math

import pytest

from jspace_binding.analysis.binding_score import (
    collect_scores,
    control_binding_score,
    family_binding_score,
    family_gap_changes,
    position_average,
)
from jspace_binding.types import (
    Construction,
    EditType,
    InjectionSite,
    Position,
    ProbeKind,
    PushSign,
    Role,
    TrialResult,
)

PUSH, NO_EDIT = EditType.ROLE_PUSH, EditType.NO_EDIT
NULL, RANDOM = EditType.NULL_NON_PARTICIPANT, EditType.RANDOM_DIRECTION
SHUFFLED = EditType.SHUFFLED_LABEL_DIRECTION
TO_A, TO_P = PushSign.TOWARD_AGENT, PushSign.TOWARD_PATIENT


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


L_AGENT = math.log(0.75 / 0.25)  # entity's answer logit when agent
L_PATIENT = math.log(0.05 / 0.95)  # ... when patient

# Logit tables per condition: {(edit, sign): (l_agent_cond, l_patient_cond)}.
# BINDING: the push interacts with the assigned role (+2.5 against the
# assignment, +0.3 with it) -> dG(+) = dG(-) = -2.2 -> BS = 2.2.
BINDING = {
    (NO_EDIT, None): (L_AGENT, L_PATIENT),
    (PUSH, TO_A): (L_AGENT + 0.3, L_PATIENT + 2.5),
    (PUSH, TO_P): (L_AGENT - 2.5, L_PATIENT - 0.3),
}
# BAG: a uniform +/-0.8 log-odds shift regardless of role -> BS = 0.
BAG = {
    (NO_EDIT, None): (L_AGENT, L_PATIENT),
    (PUSH, TO_A): (L_AGENT + 0.8, L_PATIENT + 0.8),
    (PUSH, TO_P): (L_AGENT - 0.8, L_PATIENT - 0.8),
}
# Controls sitting at baseline under both signs.
CONTROLS_FLAT = {
    (NULL, TO_A): (L_AGENT, L_PATIENT),
    (NULL, TO_P): (L_AGENT, L_PATIENT),
    (RANDOM, TO_A): (L_AGENT, L_PATIENT),
    (RANDOM, TO_P): (L_AGENT, L_PATIENT),
    (SHUFFLED, TO_A): (L_AGENT, L_PATIENT),
    (SHUFFLED, TO_P): (L_AGENT, L_PATIENT),
}


def _trial(
    role: Role,
    position: Position,
    edit_type: EditType,
    sign: PushSign | None,
    p_entity: float,
    *,
    probe: ProbeKind = ProbeKind.ROLE,
    site: InjectionSite = InjectionSite.FINAL_TOKEN,
    family_id: str = "fam-0",
    pair_id: str = "doctor->nurse",
    construction: Construction = Construction.ACTIVE_PASSIVE,
) -> TrialResult:
    entity, counterpart = pair_id.split("->", 1)
    return TrialResult(
        family_id=family_id,
        pair_id=pair_id,
        construction=construction,
        role=role,
        position=position,
        edit_type=edit_type,
        probe_kind=probe,
        injection_site=site,
        answer_probs={entity: p_entity, counterpart: 0.01, "lawyer": 0.30},
        push_sign=sign,
    )


def _family_trials(
    logit_table: dict[tuple[EditType, PushSign | None], tuple[float, float]],
    bias: float = 0.0,
    **kwargs: object,
) -> list[TrialResult]:
    """One ROLE-probe trial per (role, position, edit, sign); optional
    FIRST-position logit bias; probabilities are sigmoids of the table."""
    trials = []
    for (edit_type, sign), (l_agent, l_patient) in logit_table.items():
        for role, logit_value in ((Role.AGENT, l_agent), (Role.PATIENT, l_patient)):
            for position in Position:
                l_cell = logit_value + (bias if position is Position.FIRST else 0.0)
                trials.append(_trial(role, position, edit_type, sign, _sigmoid(l_cell), **kwargs))
    return trials


def test_position_average_collapses_positions() -> None:
    values = {
        (Role.AGENT, Position.FIRST): 0.6,
        (Role.AGENT, Position.SECOND): 0.5,
        (Role.PATIENT, Position.FIRST): 0.2,
        (Role.PATIENT, Position.SECOND): 0.1,
    }
    assert position_average(values) == pytest.approx({Role.AGENT: 0.55, Role.PATIENT: 0.15})


def test_binding_crossover_recovers_hand_computed_score() -> None:
    trials = _family_trials(BINDING)
    assert family_binding_score(trials, "doctor") == pytest.approx(2.2, abs=1e-9)
    gaps = family_gap_changes(trials, "doctor")
    assert gaps[TO_A] == pytest.approx(-2.2, abs=1e-9)
    assert gaps[TO_P] == pytest.approx(-2.2, abs=1e-9)


def test_bag_uniform_logit_shift_scores_zero() -> None:
    assert family_binding_score(_family_trials(BAG), "doctor") == pytest.approx(0.0, abs=1e-9)


def test_bag_would_look_like_binding_in_raw_probability() -> None:
    """The floor/ceiling artifact the log-odds readout removes: under BAG the
    raw-probability gap DOES change (the near-ceiling agent condition moves
    less than the mid-range patient condition), while the log-odds score is 0.
    """
    p = {key: tuple(_sigmoid(x) for x in pair) for key, pair in BAG.items()}
    raw_gap_change = (p[(PUSH, TO_A)][0] - p[(PUSH, TO_A)][1]) - (
        p[(NO_EDIT, None)][0] - p[(NO_EDIT, None)][1]
    )
    assert abs(raw_gap_change) > 0.05  # raw probability: a phantom "effect"
    assert family_binding_score(_family_trials(BAG), "doctor") == pytest.approx(0.0, abs=1e-9)


def test_position_bias_cancels_under_averaging() -> None:
    for levels in (BINDING, BAG):
        unbiased = family_binding_score(_family_trials(levels), "doctor")
        biased = family_binding_score(_family_trials(levels, bias=0.25), "doctor")
        assert biased == pytest.approx(unbiased, abs=1e-9)


def test_neutral_probe_trials_are_ignored() -> None:
    trials = _family_trials(BINDING) + [
        _trial(role, position, EditType.IDENTITY_SWAP, None, 0.99, probe=ProbeKind.NEUTRAL)
        for role in Role
        for position in Position
    ]
    assert family_binding_score(trials, "doctor") == pytest.approx(2.2, abs=1e-9)


def test_control_binding_score_runs_the_same_statistic() -> None:
    trials = _family_trials({(NO_EDIT, None): (L_AGENT, L_PATIENT), **CONTROLS_FLAT})
    for control in (NULL, RANDOM, SHUFFLED):
        assert control_binding_score(trials, "doctor", control) == pytest.approx(0.0, abs=1e-9)


def test_control_binding_score_rejects_non_control_edits() -> None:
    trials = _family_trials(BINDING)
    with pytest.raises(ValueError, match="not a control edit"):
        control_binding_score(trials, "doctor", EditType.ROLE_PUSH)


def test_missing_sign_cell_raises_by_name() -> None:
    incomplete = {k: v for k, v in BINDING.items() if k != (PUSH, TO_P)}
    with pytest.raises(ValueError, match="toward_patient"):
        family_binding_score(_family_trials(incomplete), "doctor")


def test_collect_scores_groups_by_construction_and_pair() -> None:
    fam_a = _family_trials({**BINDING, **CONTROLS_FLAT})
    fam_b = _family_trials(
        {**BAG, **CONTROLS_FLAT},
        family_id="fam-1",
        pair_id="teacher->student",
        construction=Construction.DATIVE,
    )
    # The same family re-measured at the entity-token site must not contaminate
    # (or, via the duplicate-cell guard, break) the final-token table.
    entity_site = [
        _trial(role, position, PUSH, sign, 0.99, site=InjectionSite.ENTITY_TOKEN)
        for role in Role
        for position in Position
        for sign in PushSign
    ]
    table = collect_scores(fam_a + fam_b + entity_site, InjectionSite.FINAL_TOKEN)
    assert table.real == {
        ("active_passive", "doctor->nurse"): [pytest.approx(2.2, abs=1e-9)],
        ("dative", "teacher->student"): [pytest.approx(0.0, abs=1e-9)],
    }
    # 3 controls x 2 families sample the null band; all flat here.
    assert table.null_band == pytest.approx([0.0] * 6, abs=1e-9)
    # Per-sign descriptive breakdown pools across families.
    assert table.gap_changes[TO_A.value] == pytest.approx([-2.2, 0.0], abs=1e-9)
    assert table.gap_changes[TO_P.value] == pytest.approx([-2.2, 0.0], abs=1e-9)
