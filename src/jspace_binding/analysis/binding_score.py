"""Binding score: position-averaged difference-in-differences per ItemFamily.

Pure numpy — no model dependencies — so the math heart is unit-testable on any
machine. Probabilities arrive as TrialResult records; this module only reads
answer_probs[target] and never renormalizes (full-softmax reads, per protocol).

Notation (proposal): p(role, edit) is P(target token) at the ROLE probe,
averaged over the two surface positions of the target concept, and

    BS_i = [p(agent, real) - p(patient, real)]
         - [p(agent, no_edit) - p(patient, no_edit)]

A binding representation moves mass to the target only when the swapped concept
occupies the probed role, so BS_i >> 0. A bag-of-concepts representation moves
mass role-independently, both brackets match, and BS_i ~ 0. Control edits
(null_non_participant, random_direction) plug into the same DiD in place of
REAL and form the null band the real scores must clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jspace_binding.types import (
    EditType,
    InjectionSite,
    Position,
    ProbeKind,
    Role,
    TrialResult,
)

CONTROL_EDIT_TYPES: tuple[EditType, ...] = (
    EditType.NULL_NON_PARTICIPANT,
    EditType.RANDOM_DIRECTION,
)


def position_average(probs: dict[tuple[Role, Position], float]) -> dict[Role, float]:
    """Collapse the position nuisance axis: p(role) = mean over FIRST/SECOND.

    Because the target concept appears once in each position per role, any
    surface-order bias (e.g. earlier tokens easier to recall) contributes
    equally to p(agent) and p(patient) and cancels in the DiD. A role present
    with only one position would silently reintroduce that confound, so it
    raises instead.
    """
    out: dict[Role, float] = {}
    for role in Role:
        present = [pos for pos in Position if (role, pos) in probs]
        if not present:
            continue
        if len(present) < len(Position):
            missing = next(pos for pos in Position if (role, pos) not in probs)
            raise ValueError(
                f"position_average: role {role.value!r} has no {missing.value!r} entry; "
                "averaging one position would reintroduce the word-order confound"
            )
        out[role] = float(np.mean([probs[(role, pos)] for pos in Position]))
    return out


def _did_score(trials: list[TrialResult], target_token: str, treatment: EditType) -> float:
    """Shared DiD core: [p(agent, treatment) - p(patient, treatment)]
    - [p(agent, no_edit) - p(patient, no_edit)], each p position-averaged.

    Requires exactly one ROLE-probe trial per (role, position, edit) cell for
    edit in {treatment, NO_EDIT} — 8 cells total. Missing or duplicate cells
    raise ValueError naming the cell, never a silently wrong score.
    """
    wanted = (treatment, EditType.NO_EDIT)
    probs: dict[tuple[Role, Position, EditType], float] = {}
    for t in trials:
        if t.probe_kind is not ProbeKind.ROLE or t.edit_type not in wanted:
            continue
        cell = (t.role, t.position, t.edit_type)
        cell_name = f"({t.role.value}, {t.position.value}, {t.edit_type.value})"
        if cell in probs:
            raise ValueError(
                f"duplicate trial for cell {cell_name} in family {t.family_id!r}; "
                "pass one family's trials at one injection site"
            )
        if target_token not in t.answer_probs:
            raise ValueError(
                f"target token {target_token!r} missing from answer_probs of cell "
                f"{cell_name} in family {t.family_id!r}"
            )
        probs[cell] = t.answer_probs[target_token]

    missing_cells = [
        f"({role.value}, {position.value}, {edit.value})"
        for edit in wanted
        for role in Role
        for position in Position
        if (role, position, edit) not in probs
    ]
    if missing_cells:
        raise ValueError(
            f"cannot compute DiD vs {treatment.value!r}: missing ROLE-probe cells "
            + ", ".join(missing_cells)
        )

    p = {
        edit: position_average(
            {(role, pos): probs[(role, pos, edit)] for role in Role for pos in Position}
        )
        for edit in wanted
    }
    treated, baseline = p[treatment], p[EditType.NO_EDIT]
    return float(
        (treated[Role.AGENT] - treated[Role.PATIENT])
        - (baseline[Role.AGENT] - baseline[Role.PATIENT])
    )


def family_binding_score(trials: list[TrialResult], target_token: str) -> float:
    """Binding score of one family from its ROLE-probe trials at one site.

    BS_i = [p(agent, real) - p(patient, real)]
         - [p(agent, no_edit) - p(patient, no_edit)],
    each p = P(target_token) position-averaged (see position_average). Raises
    ValueError, naming the cell, if any of the 8 required
    (role, position, edit in {real, no_edit}) cells is absent or duplicated.
    Primary analysis reads FINAL_TOKEN-site trials; site filtering is the
    caller's job (collect_scores) so the same math serves both sites.
    """
    return _did_score(trials, target_token, treatment=EditType.REAL)


def control_binding_score(
    trials: list[TrialResult], target_token: str, edit_type: EditType
) -> float:
    """Same DiD with REAL replaced by a control edit; samples the null band.

    BS_i^ctrl = [p(agent, ctrl) - p(patient, ctrl)]
              - [p(agent, no_edit) - p(patient, no_edit)].
    The control carries no role information, so under any representation its
    score should sit near 0 — the empirical distribution of these scores is
    the band a genuine binding effect must clear.
    """
    if edit_type not in CONTROL_EDIT_TYPES:
        allowed = ", ".join(e.value for e in CONTROL_EDIT_TYPES)
        raise ValueError(
            f"control_binding_score: {edit_type.value!r} is not a control edit ({allowed})"
        )
    return _did_score(trials, target_token, treatment=edit_type)


@dataclass
class ScoreTable:
    """Family-level scores grouped for stats and the forest plot.

    real maps group_key = (construction.value, pair_id) to that group's
    per-family BS_i, keeping constructions separate so a single-construction
    effect cannot masquerade as general binding. null_band pools control-edit
    scores across all groups: controls estimate the same DiD with no role
    information, so they share one null distribution.
    """

    real: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    null_band: list[float] = field(default_factory=list)


def collect_scores(all_trials: list[TrialResult], site: InjectionSite) -> ScoreTable:
    """Compute every family's real and control binding scores at one site.

    Filters to ROLE-probe trials at `site`, groups by family, and computes
    family_binding_score wherever REAL trials exist plus one
    control_binding_score per control edit present. The target token is
    recovered from pair_id ("source->target", see ConceptPair.pair_id) since
    TrialResult does not carry the AnswerSet. Incomplete cells inside any
    attempted score raise (via _did_score) rather than being skipped.
    """
    by_family: dict[str, list[TrialResult]] = {}
    for t in all_trials:
        if t.probe_kind is ProbeKind.ROLE and t.injection_site is site:
            by_family.setdefault(t.family_id, []).append(t)

    table = ScoreTable()
    for trials in by_family.values():
        first = trials[0]
        target_token = first.pair_id.split("->", 1)[1]
        edits_present = {t.edit_type for t in trials}
        if EditType.REAL in edits_present:
            group_key = (first.construction.value, first.pair_id)
            table.real.setdefault(group_key, []).append(
                family_binding_score(trials, target_token)
            )
        for control in CONTROL_EDIT_TYPES:
            if control in edits_present:
                table.null_band.append(control_binding_score(trials, target_token, control))
    return table
