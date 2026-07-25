"""The dative RECIPIENT probe and its sign convention.

The dative's ROLE probe asks who the GIVER is, so the recipient is never
queried. ProbeKind.RECIPIENT adds that readout. Its polarity is inverted —
P(entity) rises as the entity becomes more PATIENT-like — so
binding_score.PROBE_ORIENTATION negates it to keep "positive = binding".

That negation is the whole risk of this feature: get it wrong and a real
effect reports as a clean negative. The load-bearing test is
test_recipient_probe_scores_same_magnitude_as_role_probe, which measures the
SAME planted binding effect from both sides and requires the same answer.
"""

from __future__ import annotations

import math

import pytest

from jspace_binding.analysis.binding_score import (
    PROBE_ORIENTATION,
    collect_scores,
    family_binding_score,
)
from jspace_binding.stimuli.templates import build_family
from jspace_binding.types import (
    ConceptPair,
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
TO_A, TO_P = PushSign.TOWARD_AGENT, PushSign.TOWARD_PATIENT

L_AGENT = math.log(0.75 / 0.25)
L_PATIENT = math.log(0.05 / 0.95)

# Planted binding effect, as seen through the ROLE probe (entity is probable
# when it is the AGENT). dG = -2.2 for both signs -> BS = +2.2.
ROLE_BINDING = {
    (NO_EDIT, None): (L_AGENT, L_PATIENT),
    (PUSH, TO_A): (L_AGENT + 0.3, L_PATIENT + 2.5),
    (PUSH, TO_P): (L_AGENT - 2.5, L_PATIENT - 0.3),
}
# The SAME effect through the RECIPIENT probe: the entity is probable when it
# is the PATIENT (= recipient), and every deviation mirrors. Raw crossover is
# -2.2; PROBE_ORIENTATION restores +2.2.
RECIPIENT_BINDING = {
    (NO_EDIT, None): (L_PATIENT, L_AGENT),
    (PUSH, TO_A): (L_PATIENT - 0.3, L_AGENT - 2.5),
    (PUSH, TO_P): (L_PATIENT + 2.5, L_AGENT + 0.3),
}
# Bag workspace seen through the recipient probe: uniform shift, no crossover.
RECIPIENT_BAG = {
    (NO_EDIT, None): (L_PATIENT, L_AGENT),
    (PUSH, TO_A): (L_PATIENT + 0.8, L_AGENT + 0.8),
    (PUSH, TO_P): (L_PATIENT - 0.8, L_AGENT - 0.8),
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _family_trials(
    logit_table: dict[tuple[EditType, PushSign | None], tuple[float, float]],
    probe: ProbeKind,
    construction: Construction = Construction.DATIVE,
    family_id: str = "fam-0",
) -> list[TrialResult]:
    trials = []
    for (edit_type, sign), (l_agent, l_patient) in logit_table.items():
        for role, value in ((Role.AGENT, l_agent), (Role.PATIENT, l_patient)):
            for position in Position:
                trials.append(
                    TrialResult(
                        family_id=family_id,
                        pair_id="doctor->nurse",
                        construction=construction,
                        role=role,
                        position=position,
                        edit_type=edit_type,
                        probe_kind=probe,
                        injection_site=InjectionSite.FINAL_TOKEN,
                        answer_probs={
                            "doctor": _sigmoid(value),
                            "nurse": 0.01,
                            "lawyer": 0.30,
                        },
                        push_sign=sign,
                    )
                )
    return trials


# ------------------------------------------------------------------ #
# The sign convention                                                 #
# ------------------------------------------------------------------ #


def test_recipient_probe_scores_same_magnitude_as_role_probe() -> None:
    """One planted effect, measured from both sides, must agree.

    If PROBE_ORIENTATION were dropped, this returns -2.2 and a real binding
    effect would be reported as a clean negative.
    """
    from_role = family_binding_score(
        _family_trials(ROLE_BINDING, ProbeKind.ROLE), "doctor", ProbeKind.ROLE
    )
    from_recipient = family_binding_score(
        _family_trials(RECIPIENT_BINDING, ProbeKind.RECIPIENT), "doctor", ProbeKind.RECIPIENT
    )

    assert from_role == pytest.approx(2.2, abs=1e-9)
    assert from_recipient == pytest.approx(2.2, abs=1e-9)


def test_recipient_orientation_is_the_inverting_one() -> None:
    assert PROBE_ORIENTATION[ProbeKind.ROLE] == 1.0
    assert PROBE_ORIENTATION[ProbeKind.RECIPIENT] == -1.0


def test_recipient_probe_bag_lands_at_zero() -> None:
    """A role-blind uniform push still cancels; the flip cannot manufacture one."""
    score = family_binding_score(
        _family_trials(RECIPIENT_BAG, ProbeKind.RECIPIENT), "doctor", ProbeKind.RECIPIENT
    )
    assert score == pytest.approx(0.0, abs=1e-9)


def test_role_and_recipient_trials_are_scored_separately() -> None:
    """Mixed trials must not bleed: each probe scores only its own cells."""
    mixed = _family_trials(ROLE_BINDING, ProbeKind.ROLE) + _family_trials(
        RECIPIENT_BAG, ProbeKind.RECIPIENT
    )

    assert family_binding_score(mixed, "doctor", ProbeKind.ROLE) == pytest.approx(2.2, abs=1e-9)
    assert family_binding_score(mixed, "doctor", ProbeKind.RECIPIENT) == pytest.approx(
        0.0, abs=1e-9
    )


def test_collect_scores_selects_by_probe_kind() -> None:
    mixed = _family_trials(ROLE_BINDING, ProbeKind.ROLE) + _family_trials(
        RECIPIENT_BINDING, ProbeKind.RECIPIENT
    )

    role_table = collect_scores(mixed, InjectionSite.FINAL_TOKEN, ProbeKind.ROLE)
    recipient_table = collect_scores(mixed, InjectionSite.FINAL_TOKEN, ProbeKind.RECIPIENT)

    assert role_table.real[("dative", "doctor->nurse")] == [pytest.approx(2.2, abs=1e-9)]
    assert recipient_table.real[("dative", "doctor->nurse")] == [pytest.approx(2.2, abs=1e-9)]


def test_collect_scores_recipient_is_empty_without_recipient_trials() -> None:
    role_only = _family_trials(ROLE_BINDING, ProbeKind.ROLE)
    table = collect_scores(role_only, InjectionSite.FINAL_TOKEN, ProbeKind.RECIPIENT)
    assert table.real == {}


# ------------------------------------------------------------------ #
# Which families carry the probe                                      #
# ------------------------------------------------------------------ #


def _family(construction: Construction):
    return build_family(
        pair=ConceptPair("doctor", "nurse"),
        construction=construction,
        other_entity="lawyer",
        verb_lemma="handed" if construction is Construction.DATIVE else "treated",
        family_index=0,
    )


def test_only_the_dative_carries_a_recipient_probe() -> None:
    for construction in Construction:
        family = _family(construction)
        if construction is Construction.DATIVE:
            assert family.recipient_probe
        else:
            assert family.recipient_probe == ""


def test_dative_recipient_probe_wording() -> None:
    family = _family(Construction.DATIVE)
    assert family.recipient_probe == (
        "Question: Who was handed the letter by someone? Answer: The"
    )
    # It must query the OTHER participant than the role probe.
    assert family.role_probe == "Question: Who handed a letter to someone? Answer: The"
    assert family.recipient_probe != family.role_probe


def test_recipient_probe_is_entity_free() -> None:
    """Same invariant the other probes hold: no participant named in the probe."""
    family = _family(Construction.DATIVE)
    for word in ("doctor", "nurse", "lawyer"):
        assert word not in family.recipient_probe


# ------------------------------------------------------------------ #
# Ground-truth recovery through the dummy backend                     #
# ------------------------------------------------------------------ #


def _dative_scores(mode: str) -> dict[ProbeKind, float]:
    """Run the dummy end-to-end and score the dative from both probes."""
    import statistics
    import tempfile
    from dataclasses import replace
    from pathlib import Path

    from jspace_binding.config import Config
    from jspace_binding.experiments.primary import run_primary
    from jspace_binding.model.dummy import DummyModel
    from jspace_binding.stimuli.generate import generate_families

    config = Config.from_yaml("configs/ci.yaml")
    config = replace(
        config, paths=replace(config.paths, results=Path(tempfile.mkdtemp()))
    )
    trials = run_primary(config, DummyModel(mode=mode, seed=0), generate_families(config))
    out = {}
    for probe_kind in (ProbeKind.ROLE, ProbeKind.RECIPIENT):
        table = collect_scores(trials, InjectionSite.FINAL_TOKEN, probe_kind)
        out[probe_kind] = statistics.mean(table.real[("dative", "doctor->nurse")])
    return out


def test_dummy_models_the_recipient_probe() -> None:
    """The dummy must plant the mirrored ground truth for the recipient probe.

    It classifies probes by wording, and "Who was handed..." matches its
    role-probe test — so without the readout swap in answer_distribution it
    plants ROLE ground truth, PROBE_ORIENTATION negates it, and binding mode
    reports a large spurious NEGATIVE dative score that looks like a finding.
    Both probes must recover the same planted ~+2.2.
    """
    scores = _dative_scores("binding")
    assert scores[ProbeKind.ROLE] > 1.5
    assert scores[ProbeKind.RECIPIENT] > 1.5, (
        "recipient probe failed to recover the planted binding effect "
        f"(got {scores[ProbeKind.RECIPIENT]:+.3f}); a negative here means the "
        "dummy is not modelling the probe's inverted polarity"
    )


def test_dummy_bag_mode_is_null_at_both_probes() -> None:
    """A role-blind push cancels at both readouts; neither flip invents one."""
    scores = _dative_scores("bag")
    assert abs(scores[ProbeKind.ROLE]) < 0.5
    assert abs(scores[ProbeKind.RECIPIENT]) < 0.5
