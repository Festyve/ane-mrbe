"""Edit planning: (EditType, PushSign) -> backend-agnostic EditSpec.

Pure data plumbing; the residual-stream math lives in the model backend. The
lens artifact is a per-layer averaged Jacobian `J_l`; the J-lens vector for
vocabulary token t is `v_t = J_l^T W_U[t]`, and an activation's J-space
component is recovered by sparse pursuit against those vectors (Gurnee et al.
2026 §2). Both edit families below are defined at the injection-site token.

Role push (primary). With `r` the fitted unit role direction for the spec's
entity at the site (difference-of-means of J-space components, directions.fit):

    h_patched = h + s * c * unit(r)

`s` is +1 for TOWARD_AGENT and -1 for TOWARD_PATIENT; `c` is calibrated on the
fitting corpus, never on the primary stimuli. unit() keeps `c` in residual-norm
units, so every push perturbs the stream by exactly `c`. The same (r, s, c,
site) applies to every sentence in a condition — byte-identical across the
minimal pair, never conditioned on the sentence's own role label, which is what
licenses reading a role-dependent effect as binding. The three strength-matched
controls share these mechanics: NULL_NON_PARTICIPANT pushes an entity absent
from the sentence, RANDOM_DIRECTION a seeded random unit direction, and
SHUFFLED_LABEL_DIRECTION the same entity's shuffled-label refit.

Identity swap (control only — intervention-strength check, Gurnee et al. §2.5).
With `V = [v_s v_t]` the J-lens vectors of the swap tokens:

    c = pinv(V) @ h          # c[0] source amount, c[1] target amount
    h_patched = h + V @ (sigma(c) - c)

`sigma` swaps the two coordinates, optionally scaled by alpha (None = pure
swap); the component orthogonal to span(V) is untouched. The source paper
applies its swap at all token positions where we anchor at the injection site —
a deliberate difference.

RQ2 ablations are built directly by experiments.rq2_ablation: ABLATE_JSPACE
zeroes h's projection onto the top-`ablate_k` active J-lens vectors (§3.5.2),
and ABLATE_RANDOM_SUBSPACE removes a seeded random orthonormal basis with the
same number of columns — the capacity-matched comparison.
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
