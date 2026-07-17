"""Steering-strength calibration (proposal, Methods / Concept-swap intervention).

Two separate dials, calibrated on separate data, never on the primary
stimulus set:

- push_coefficient (ROLE_PUSH and its controls): smallest grid value whose
  toward-agent push on PATIENT-role FITTING-CORPUS sentences lifts the
  entity's role-probe answer by at least min_logit_shift. Calibrating on the
  fitting corpus keeps calibration and testing on disjoint data.
- alpha (IDENTITY_SWAP): smallest grid value whose swap detectably moves the
  role-neutral readout on a small stimulus sample — P(counterpart) rises and
  P(entity) falls at the NEUTRAL probe. This is the "smallest value that
  works" rule: strong enough to be interpretable, not so strong it saturates
  the residual stream.

Both searches are written against the WorkspaceModel protocol. The DummyModel
has no strength dial (it ignores coefficient/alpha), so a --dry-run
calibration returns the smallest grid value — the scripts exist to be
exercised end-to-end, the numbers only mean something on the real backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from jspace_binding.analysis.binding_score import logit
from jspace_binding.model.base import WorkspaceModel
from jspace_binding.stimuli.fitting_corpus import FittingExample
from jspace_binding.types import (
    EditSpec,
    EditType,
    InjectionSite,
    ItemFamily,
    PushSign,
    Role,
)

_NO_EDIT = EditSpec(edit_type=EditType.NO_EDIT)


@dataclass(frozen=True)
class CalibrationResult:
    value: float
    grid: tuple[float, ...]
    shifts: dict[float, float]  # grid value -> mean observed shift
    criterion: str


def calibrate_push_coefficient(
    model: WorkspaceModel,
    examples: list[FittingExample],
    grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    min_logit_shift: float = 0.5,
    site: InjectionSite = InjectionSite.FINAL_TOKEN,
    max_examples: int = 50,
) -> CalibrationResult:
    """Smallest coefficient whose toward-agent push moves patient-role fitting
    sentences by >= min_logit_shift in the entity's role-probe log-odds."""
    patients = [ex for ex in examples if ex.role is Role.PATIENT][:max_examples]
    if not patients:
        raise ValueError("calibrate_push_coefficient: no patient-role fitting examples")
    shifts: dict[float, float] = {}
    chosen: float | None = None
    for coefficient in sorted(grid):
        total = 0.0
        for ex in patients:
            push = EditSpec(
                edit_type=EditType.ROLE_PUSH,
                entity=ex.entity,
                sign=PushSign.TOWARD_AGENT,
                coefficient=coefficient,
            )
            tokens = (ex.entity, ex.other)
            pushed = model.answer_distribution(ex.sentence, ex.role_probe, push, site, tokens)
            base = model.answer_distribution(ex.sentence, ex.role_probe, _NO_EDIT, site, tokens)
            total += logit(pushed[ex.entity]) - logit(base[ex.entity])
        shifts[coefficient] = total / len(patients)
        if chosen is None and shifts[coefficient] >= min_logit_shift:
            chosen = coefficient
    if chosen is None:
        raise ValueError(
            f"no grid value reached min_logit_shift={min_logit_shift}; observed {shifts}. "
            "Either extend the grid or treat this as an intervention-strength failure."
        )
    return CalibrationResult(
        value=chosen,
        grid=tuple(sorted(grid)),
        shifts=shifts,
        criterion=f"mean entity logit shift >= {min_logit_shift} on patient-role "
        "fitting sentences under the toward-agent push",
    )


def calibrate_identity_alpha(
    model: WorkspaceModel,
    families: list[ItemFamily],
    grid: tuple[float, ...] = (0.5, 1.0, 2.0),
    min_prob_shift: float = 0.05,
    site: InjectionSite = InjectionSite.FINAL_TOKEN,
    max_families: int = 25,
) -> CalibrationResult:
    """Smallest alpha whose identity swap moves the NEUTRAL-probe readout:
    P(counterpart) up by >= min_prob_shift and P(entity) down."""
    sample = families[:max_families]
    if not sample:
        raise ValueError("calibrate_identity_alpha: no families provided")
    shifts: dict[float, float] = {}
    chosen: float | None = None
    for alpha in sorted(grid):
        counterpart_shift = 0.0
        entity_shift = 0.0
        n = 0
        for family in sample:
            answers = family.answer_set
            if answers is None:
                raise ValueError(f"family {family.family_id!r} has no answer_set")
            swap = EditSpec(
                edit_type=EditType.IDENTITY_SWAP,
                swap_source=answers.entity,
                swap_target=answers.counterpart,
                alpha=alpha,
            )
            for role in Role:
                # One cell per role suffices for a strength read; FIRST position.
                stimulus = family.cells[f"{role.value}:first"]
                swapped = model.answer_distribution(
                    stimulus.sentence, family.neutral_probe, swap, site, answers.tokens
                )
                base = model.answer_distribution(
                    stimulus.sentence, family.neutral_probe, _NO_EDIT, site, answers.tokens
                )
                counterpart_shift += swapped[answers.counterpart] - base[answers.counterpart]
                entity_shift += swapped[answers.entity] - base[answers.entity]
                n += 1
        counterpart_shift /= n
        entity_shift /= n
        shifts[alpha] = counterpart_shift
        if chosen is None and counterpart_shift >= min_prob_shift and entity_shift < 0.0:
            chosen = alpha
    if chosen is None:
        raise ValueError(
            f"no grid value moved the neutral readout by >= {min_prob_shift} "
            f"(with P(entity) falling); observed counterpart shifts {shifts}. "
            "This is the intervention-strength failure mode the proposal flags: "
            "a null binding result would be uninterpretable at these strengths."
        )
    return CalibrationResult(
        value=chosen,
        grid=tuple(sorted(grid)),
        shifts=shifts,
        criterion=f"mean P(counterpart) shift >= {min_prob_shift} and P(entity) "
        "falling at the NEUTRAL probe",
    )
