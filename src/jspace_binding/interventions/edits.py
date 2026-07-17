"""Edit planning: (EditType, PushSign) -> backend-agnostic EditSpec.

This module is pure data-plumbing (stdlib only); the residual-stream math lives
in the model backend. This docstring is the canonical in-repo description of
the procedures every backend must implement.

Role push (PRIMARY — proposal, Methods / Concept-swap intervention)
-------------------------------------------------------------------
Let r be the fitted unit role direction for the spec's entity at the injection
site (directions.fit; fitted per site), expressed in J-space coordinates, and
B the lens map from J-space coordinates back to the residual stream. At the
site token:

    h_patched = h + s * c * (B @ r)

where s = +1 for PushSign.TOWARD_AGENT, -1 for TOWARD_PATIENT, and c is the
push coefficient calibrated on the fitting corpus (never on the primary
stimuli). The same (r, s, c, site) is applied to every sentence in a condition
— byte-identical across the minimal pair, never conditioned on the sentence's
own role label. That uniformity is what licenses reading any role-dependent
effect as binding.

Controls sharing the push mechanics (all strength-matched via the same c):
- NULL_NON_PARTICIPANT: push the fitted r of an entity ABSENT from the
  sentence (plan_edit picks it per family). Ties any effect to the sentence's
  relational content rather than generic workspace perturbation.
- RANDOM_DIRECTION: replace r with a seeded random unit direction in the same
  subspace — norm-matched by construction since pushes always use unit
  directions. Direction-specificity control.
- SHUFFLED_LABEL_DIRECTION: push the same entity's shuffled-label refit
  (directions.fit.shuffled_label_direction). Direction-overfitting control.

Identity swap (CONTROL ONLY — intervention-strength check)
----------------------------------------------------------
The Gurnee et al. coordinate swap, inherited wholesale, paired exclusively
with the NEUTRAL probe. Let v_s and v_t be the lens identity directions for
swap_source and swap_target, V = [v_s v_t], and h the residual activation:

    c = pinv(V) @ h                      # c[0]: source amount, c[1]: target amount
    h_patched = h + V @ (sigma(c) - c)

where sigma swaps the two coordinates, optionally scaled by alpha (None = 1.0,
a pure swap). If this fails to move the role-neutral readout, the lens is too
weak for any null role result to be interpretable.

NO_EDIT: the untouched baseline the difference-in-differences subtracts.
"""

from __future__ import annotations

from jspace_binding.types import EditSpec, EditType, ItemFamily, PushSign


def choose_non_participant(family: ItemFamily, candidates: tuple[str, ...]) -> str:
    """First candidate entity that does not occur in the family's sentences.

    The pushed direction must belong to a concept with no role in the scene;
    a candidate matching the entity, counterpart, or other participant would
    silently turn the control into a (weaker) real edit.
    """
    occupied = {
        family.concept_pair.entity,
        family.concept_pair.counterpart,
        family.other_entity,
    }
    for candidate in candidates:
        if candidate not in occupied:
            return candidate
    raise ValueError(
        f"no non-participant candidate outside {sorted(occupied)} for family "
        f"{family.family_id!r}; extend stimuli.vocab.NON_PARTICIPANT_CANDIDATES"
    )


def plan_edit(
    family: ItemFamily,
    edit_type: EditType,
    sign: PushSign | None,
    non_participant_candidates: tuple[str, ...],
    alpha: float | None,
    coefficient: float | None,
    seed: int,
) -> EditSpec:
    """Translate an (EditType, PushSign) condition into the EditSpec a backend executes.

    Push edits require a sign and share the same coefficient so controls are
    strength-matched to the real push; IDENTITY_SWAP and NO_EDIT take no sign.
    seed is recorded only for RANDOM_DIRECTION, the sole stochastic edit.
    """
    is_push = edit_type in (
        EditType.ROLE_PUSH,
        EditType.NULL_NON_PARTICIPANT,
        EditType.RANDOM_DIRECTION,
        EditType.SHUFFLED_LABEL_DIRECTION,
    )
    if is_push and sign is None:
        raise ValueError(f"{edit_type.value} is a direction push and requires a PushSign")
    if not is_push and sign is not None:
        raise ValueError(f"{edit_type.value} takes no PushSign, got {sign.value}")

    match edit_type:
        case EditType.ROLE_PUSH:
            return EditSpec(
                edit_type=edit_type,
                entity=family.concept_pair.entity,
                sign=sign,
                coefficient=coefficient,
            )
        case EditType.SHUFFLED_LABEL_DIRECTION:
            return EditSpec(
                edit_type=edit_type,
                entity=family.concept_pair.entity,
                sign=sign,
                coefficient=coefficient,
            )
        case EditType.NULL_NON_PARTICIPANT:
            return EditSpec(
                edit_type=edit_type,
                entity=choose_non_participant(family, non_participant_candidates),
                sign=sign,
                coefficient=coefficient,
            )
        case EditType.RANDOM_DIRECTION:
            return EditSpec(
                edit_type=edit_type,
                sign=sign,
                coefficient=coefficient,
                seed=seed,
            )
        case EditType.IDENTITY_SWAP:
            return EditSpec(
                edit_type=edit_type,
                swap_source=family.concept_pair.entity,
                swap_target=family.concept_pair.counterpart,
                alpha=alpha,
            )
        case EditType.NO_EDIT:
            return EditSpec(edit_type=EditType.NO_EDIT)
        case _:
            raise ValueError(f"Unhandled edit type: {edit_type!r}")
