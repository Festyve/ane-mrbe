"""Edit planning: (EditType, PushSign) -> backend-agnostic EditSpec.

This module is pure data-plumbing (stdlib only); the residual-stream math lives
in the model backend. This docstring is the canonical in-repo description of
the procedures every backend must implement.

Grounding (Gurnee et al. 2026, §2): the lens artifact is a per-layer averaged
Jacobian J_l; the J-LENS VECTOR for vocabulary token t at layer l is
v_t = J_l^T W_U[t], a residual-stream direction; the J-space is the set of
sparse nonnegative combinations of such vectors, and an activation's J-space
COMPONENT is recovered by sparse pursuit against them (their §2.3).

Role push (PRIMARY — proposal, Methods / Concept-swap intervention)
-------------------------------------------------------------------
Let r be the fitted unit role direction for the spec's entity at the
injection site: the difference-of-means of J-space COMPONENTS (agent minus
patient exemplars, directions.fit), which lives directly in residual space.
At the site token:

    h_patched = h + s * c * unit(r)

where s = +1 for PushSign.TOWARD_AGENT, -1 for TOWARD_PATIENT, and c is the
push coefficient calibrated on the fitting corpus (never on the primary
stimuli). unit() keeps c in residual-norm units, so every push — real or
control — perturbs the stream by exactly c. The same (r, s, c, site) is
applied to every sentence in a condition — byte-identical across the minimal
pair, never conditioned on the sentence's own role label. That uniformity is
what licenses reading any role-dependent effect as binding.

Controls sharing the push mechanics (all strength-matched at exactly c):
- NULL_NON_PARTICIPANT: push the fitted r of an entity ABSENT from the
  sentence (plan_edit picks it per family). Ties any effect to the sentence's
  relational content rather than generic workspace perturbation.
- RANDOM_DIRECTION: replace r with a seeded random unit residual-space
  direction. Direction-specificity control.
- SHUFFLED_LABEL_DIRECTION: push the same entity's shuffled-label refit
  (directions.fit.shuffled_label_direction). Direction-overfitting control.

Identity swap (CONTROL ONLY — intervention-strength check)
----------------------------------------------------------
The Gurnee et al. coordinate swap (their §2.5), inherited wholesale, paired
exclusively with the NEUTRAL probe. With v_s, v_t the J-LENS VECTORS of the
swap tokens at the layer, V = [v_s v_t], and h the residual activation:

    c = pinv(V) @ h                      # c[0]: source amount, c[1]: target amount
    h_patched = h + V @ (sigma(c) - c)

where sigma swaps the two coordinates, optionally scaled by alpha (None = 1.0,
a pure swap). The component of h orthogonal to span(V) is untouched. If this
fails to move the role-neutral readout, the lens is too weak for any null
role result to be interpretable. (Comparability note: several of the source
paper's swap experiments apply the swap at ALL token positions; our design
anchors it at the injection site — a deliberate difference, flagged in docs.)

NO_EDIT: the untouched baseline the difference-in-differences subtracts.

RQ2 ablations (never part of the primary sweep; experiments.rq2_ablation
builds their EditSpecs directly)
--------------------------------
ABLATE_JSPACE (their §3.5.2): zero h's projection onto the span of the
top-k most strongly active J-lens vectors at the site (k = model.ablate_k,
paper uses 10).
ABLATE_RANDOM_SUBSPACE: h_patched = h - Q @ (Q^T @ h) for a seeded random
orthonormal basis Q with the SAME number of columns (ablate_k) — the
capacity-matched comparison that separates "the J-space specifically" from
"any subspace of that size".
"""

from __future__ import annotations

from jspace_binding.types import (
    DIRECTION_PUSH_EDIT_TYPES,
    EditSpec,
    EditType,
    ItemFamily,
    PushSign,
)


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
    is_push = edit_type in DIRECTION_PUSH_EDIT_TYPES
    if is_push and sign is None:
        raise ValueError(f"{edit_type.value} is a direction push and requires a PushSign")
    if not is_push and sign is not None:
        raise ValueError(f"{edit_type.value} takes no PushSign, got {sign.value}")

    match edit_type:
        case EditType.ROLE_PUSH | EditType.SHUFFLED_LABEL_DIRECTION:
            # Same spec shape: the backends pick the fitted vs shuffled-label
            # variant of the entity's direction from edit_type.
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
