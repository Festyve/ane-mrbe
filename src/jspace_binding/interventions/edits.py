"""Edit planning: EditType -> backend-agnostic EditSpec.

This module is pure data-plumbing (stdlib only); the residual-stream math lives
in the model backend. This docstring is the canonical in-repo description of
the procedure every backend must implement.

Coordinate swap (Gurnee et al.)
-------------------------------
Let v_s and v_t be the lens directions for the source and target concepts, and
h the residual-stream activation at the injection site. Stack the directions
as columns, V = [v_s v_t], and read the workspace coordinates via the
pseudoinverse:

    c = pinv(V) @ h                      # c[0]: source amount, c[1]: target amount
    h_patched = h + V @ (sigma(c) - c)

where sigma swaps the two coordinates, optionally scaled by alpha:
sigma(c) = alpha * (c[1], c[0]); alpha=None means the backend default of 1.0,
a pure swap. The update lives entirely in span(V), so every component of h
orthogonal to the two concept directions is untouched — the edit exchanges
"how much source" for "how much target" and nothing else.

Controls
--------
NULL_NON_PARTICIPANT: the same procedure aimed at a concept that never appears
in the sentence (source = non-participant, target = None) — it injects/removes
a concept absent from the scene. Any probe movement it causes measures "an
edit happened here", not binding: the edit-does-anything control.

RANDOM_DIRECTION: replaces the swap delta with a random vector at the same
site, norm-matched to the REAL edit's delta and seeded for reproducibility.
If REAL and RANDOM move the probe equally, the effect is not
direction-specific: the direction-specificity control.

NO_EDIT: the untouched baseline the difference-in-differences subtracts.
"""

from __future__ import annotations

from jspace_binding.types import EditSpec, EditType, ItemFamily


def plan_edit(
    family: ItemFamily,
    edit_type: EditType,
    non_participant: str,
    alpha: float | None,
    seed: int,
) -> EditSpec:
    """Translate an EditType into the EditSpec a backend executes.

    Controls carry the same alpha as REAL so they are strength-matched; seed is
    recorded only for RANDOM_DIRECTION, the sole stochastic edit.
    """
    match edit_type:
        case EditType.REAL:
            return EditSpec(
                edit_type=EditType.REAL,
                source_concept=family.concept_pair.source,
                target_concept=family.concept_pair.target,
                alpha=alpha,
            )
        case EditType.NULL_NON_PARTICIPANT:
            return EditSpec(
                edit_type=EditType.NULL_NON_PARTICIPANT,
                source_concept=non_participant,
                target_concept=None,
                alpha=alpha,
            )
        case EditType.RANDOM_DIRECTION:
            return EditSpec(edit_type=EditType.RANDOM_DIRECTION, alpha=alpha, seed=seed)
        case EditType.NO_EDIT:
            return EditSpec(edit_type=EditType.NO_EDIT)
        case _:
            raise ValueError(f"Unhandled edit type: {edit_type!r}")
