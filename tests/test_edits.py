"""Contract tests for interventions.edits.plan_edit."""

from __future__ import annotations

import pytest

from jspace_binding.interventions.edits import choose_non_participant, plan_edit
from jspace_binding.stimuli.templates import build_family
from jspace_binding.types import ConceptPair, Construction, EditType, PushSign

CANDIDATES = ("chef", "farmer", "coach")


def _family(other: str = "lawyer"):
    return build_family(
        pair=ConceptPair("doctor", "nurse"),
        construction=Construction.ACTIVE_PASSIVE,
        other_entity=other,
        verb_lemma="treated",
        family_index=0,
    )


def test_role_push_carries_entity_sign_coefficient() -> None:
    spec = plan_edit(
        _family(), EditType.ROLE_PUSH, PushSign.TOWARD_AGENT, CANDIDATES, None, 4.0, seed=0
    )
    assert (spec.entity, spec.sign, spec.coefficient) == ("doctor", PushSign.TOWARD_AGENT, 4.0)
    assert spec.swap_source is None and spec.seed is None


def test_push_edits_require_a_sign_and_swap_takes_none() -> None:
    with pytest.raises(ValueError, match="requires a PushSign"):
        plan_edit(_family(), EditType.ROLE_PUSH, None, CANDIDATES, None, 4.0, seed=0)
    with pytest.raises(ValueError, match="takes no PushSign"):
        plan_edit(
            _family(), EditType.IDENTITY_SWAP, PushSign.TOWARD_AGENT, CANDIDATES, None, 4.0, seed=0
        )


def test_identity_swap_maps_entity_to_counterpart() -> None:
    spec = plan_edit(_family(), EditType.IDENTITY_SWAP, None, CANDIDATES, 1.5, None, seed=0)
    assert (spec.swap_source, spec.swap_target, spec.alpha) == ("doctor", "nurse", 1.5)


def test_non_participant_skips_entities_in_the_sentence() -> None:
    # "chef" is the family's other participant, so the control must fall
    # through to "farmer" — pushing a participant would corrupt the control.
    assert choose_non_participant(_family(other="chef"), CANDIDATES) == "farmer"
    spec = plan_edit(
        _family(other="chef"),
        EditType.NULL_NON_PARTICIPANT,
        PushSign.TOWARD_PATIENT,
        CANDIDATES,
        None,
        4.0,
        seed=0,
    )
    assert spec.entity == "farmer"
    with pytest.raises(ValueError, match="no non-participant candidate"):
        choose_non_participant(_family(other="chef"), ("chef", "doctor", "nurse"))


def test_random_direction_records_seed() -> None:
    spec = plan_edit(
        _family(), EditType.RANDOM_DIRECTION, PushSign.TOWARD_AGENT, CANDIDATES, None, 4.0, seed=7
    )
    assert spec.seed == 7 and spec.entity is None
