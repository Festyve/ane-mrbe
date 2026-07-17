"""Binding score: position-averaged, both-sign log-odds crossover per ItemFamily.

Pure numpy — no model dependencies — so the math heart is unit-testable on any
machine. Probabilities arrive as TrialResult records; this module reads
answer_probs[entity] (the pushed entity, NOT the identity-swap counterpart),
converts to log-odds, and never renormalizes (full-softmax reads, per
protocol).

Why log-odds (proposal, Motivation): a role-blind uniform push shifts the
entity's answer log-odds by the same increment in both role conditions
regardless of their starting point, so floor/ceiling effects cannot fake a
role interaction. In raw probability the same uniform push moves a
near-ceiling condition less than a mid-range one — exactly the artifact the
log-odds readout removes.

Notation: L(role, edit, sign) is logit P(entity) at the ROLE probe, averaged
over the two surface positions. For each push sign s the gap change is

    dG(s) = [L(agent, push_s) - L(patient, push_s)]
          - [L(agent, no_edit) - L(patient, no_edit)]

A bag workspace leaves dG(s) ~ 0 for both signs (the uniform increment cancels
in the agent-patient difference). A binding workspace shrinks the natural gap
from opposite sides — the agent-pole push lifts patient sentences most, the
patient-pole push drops agent sentences most (the crossover) — making dG(s)
negative under both signs. The family binding score folds both signs into one
positive-means-binding number:

    BS_i = -(dG(toward_agent) + dG(toward_patient)) / 2

which is algebraically the edit x role interaction summed over signs. Control
edits (null_non_participant, random_direction, shuffled_label_direction) plug
into the same statistic in place of ROLE_PUSH and form the null band the real
scores must clear. A within-family agent/patient label swap negates every
dG(s) and hence BS_i, so the sign-flip permutation test in analysis.stats
remains exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from jspace_binding.types import (
    EditType,
    InjectionSite,
    Position,
    ProbeKind,
    PushSign,
    Role,
    TrialResult,
)

CONTROL_EDIT_TYPES: tuple[EditType, ...] = (
    EditType.NULL_NON_PARTICIPANT,
    EditType.RANDOM_DIRECTION,
    EditType.SHUFFLED_LABEL_DIRECTION,
)

_EPS = 1e-9  # probability clamp so a hard 0/1 read cannot produce +/- inf


def logit(p: float) -> float:
    """log(p / (1-p)) with clamping to (_EPS, 1-_EPS)."""
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def position_average(values: dict[tuple[Role, Position], float]) -> dict[Role, float]:
    """Collapse the position nuisance axis: L(role) = mean over FIRST/SECOND.

    Because the target entity appears once in each position per role, any
    surface-order bias (e.g. earlier tokens easier to recall) contributes
    equally to L(agent) and L(patient) and cancels in the gap. A role present
    with only one position would silently reintroduce that confound, so it
    raises instead.
    """
    out: dict[Role, float] = {}
    for role in Role:
        present = [pos for pos in Position if (role, pos) in values]
        if not present:
            continue
        if len(present) < len(Position):
            missing = next(pos for pos in Position if (role, pos) not in values)
            raise ValueError(
                f"position_average: role {role.value!r} has no {missing.value!r} entry; "
                "averaging one position would reintroduce the word-order confound"
            )
        out[role] = float(np.mean([values[(role, pos)] for pos in Position]))
    return out


_CellKey = tuple[Role, Position, EditType, PushSign | None]


def _cell_logits(
    trials: list[TrialResult], entity_token: str, treatment: EditType
) -> dict[_CellKey, float]:
    """ROLE-probe logit P(entity) per (role, position, edit, sign) cell.

    Collects the treatment edit (both signs) plus NO_EDIT. Requires exactly
    one trial per cell — 12 cells total (2 roles x 2 positions x [2 signed
    treatment conditions + unsigned no-edit]). Missing or duplicate cells
    raise ValueError naming the cell, never a silently wrong score.
    """
    cells: dict[_CellKey, float] = {}
    for t in trials:
        if t.probe_kind is not ProbeKind.ROLE:
            continue
        if t.edit_type is not treatment and t.edit_type is not EditType.NO_EDIT:
            continue
        key = (t.role, t.position, t.edit_type, t.push_sign)
        name = _cell_name(key)
        if key in cells:
            raise ValueError(
                f"duplicate trial for cell {name} in family {t.family_id!r}; "
                "pass one family's trials at one injection site"
            )
        if entity_token not in t.answer_probs:
            raise ValueError(
                f"entity token {entity_token!r} missing from answer_probs of cell "
                f"{name} in family {t.family_id!r}"
            )
        cells[key] = logit(t.answer_probs[entity_token])

    wanted: list[_CellKey] = [
        (role, pos, treatment, sign) for role in Role for pos in Position for sign in PushSign
    ] + [(role, pos, EditType.NO_EDIT, None) for role in Role for pos in Position]
    missing = [_cell_name(key) for key in wanted if key not in cells]
    if missing:
        raise ValueError(
            f"cannot compute binding score vs {treatment.value!r}: missing ROLE-probe cells "
            + ", ".join(missing)
        )
    return cells


def _cell_name(key: _CellKey) -> str:
    role, pos, edit, sign = key
    sign_txt = "" if sign is None else f", {sign.value}"
    return f"({role.value}, {pos.value}, {edit.value}{sign_txt})"


def gap_change(
    trials: list[TrialResult],
    entity_token: str,
    treatment: EditType,
    sign: PushSign,
) -> float:
    """dG(sign): the edit-induced change of the position-averaged agent-patient
    log-odds gap, for one push sign of the treatment edit."""
    cells = _cell_logits(trials, entity_token, treatment)
    pushed = position_average(
        {(role, pos): cells[(role, pos, treatment, sign)] for role in Role for pos in Position}
    )
    baseline = position_average(
        {
            (role, pos): cells[(role, pos, EditType.NO_EDIT, None)]
            for role in Role
            for pos in Position
        }
    )
    return float(
        (pushed[Role.AGENT] - pushed[Role.PATIENT])
        - (baseline[Role.AGENT] - baseline[Role.PATIENT])
    )


def family_gap_changes(
    trials: list[TrialResult], entity_token: str, treatment: EditType = EditType.ROLE_PUSH
) -> dict[PushSign, float]:
    """Both signs' dG for one family — the descriptive per-sign breakdown
    (binding predicts both negative; an asymmetry, e.g. promotion works but
    demotion doesn't, shows up here rather than being averaged away)."""
    return {sign: gap_change(trials, entity_token, treatment, sign) for sign in PushSign}


def _crossover_score(trials: list[TrialResult], entity_token: str, treatment: EditType) -> float:
    gaps = family_gap_changes(trials, entity_token, treatment)
    return -(gaps[PushSign.TOWARD_AGENT] + gaps[PushSign.TOWARD_PATIENT]) / 2.0


def family_binding_score(trials: list[TrialResult], entity_token: str) -> float:
    """Binding score of one family from its ROLE-probe trials at one site.

    BS_i = -(dG(toward_agent) + dG(toward_patient)) / 2 in log-odds, positive
    under binding, ~0 under a bag workspace. Requires all 12
    (role x position x [push sign / no-edit]) cells; raises otherwise.
    Primary analysis reads FINAL_TOKEN-site trials; site filtering is the
    caller's job (collect_scores) so the same math serves both sites.
    """
    return _crossover_score(trials, entity_token, EditType.ROLE_PUSH)


def control_binding_score(
    trials: list[TrialResult], entity_token: str, edit_type: EditType
) -> float:
    """Same crossover statistic with ROLE_PUSH replaced by a control edit;
    samples the null band.

    The control carries no genuine role information for THIS sentence — the
    pushed direction belongs to an absent entity, is random, or was fit on
    shuffled labels — so under any representation its score should sit near
    0. The empirical distribution of these scores is the band a genuine
    binding effect must clear.
    """
    if edit_type not in CONTROL_EDIT_TYPES:
        allowed = ", ".join(e.value for e in CONTROL_EDIT_TYPES)
        raise ValueError(
            f"control_binding_score: {edit_type.value!r} is not a control edit ({allowed})"
        )
    return _crossover_score(trials, entity_token, edit_type)


@dataclass
class ScoreTable:
    """Family-level scores grouped for stats and the forest plot.

    real maps group_key = (construction.value, pair_id) to that group's
    per-family BS_i, keeping constructions separate so a single-construction
    effect cannot masquerade as general binding. gap_changes pools the
    per-sign dG values of the ROLE_PUSH across all families (keyed by
    PushSign.value) for the descriptive per-sign summary. null_band pools
    control-edit scores across all groups: controls estimate the same
    statistic with no role information, so they share one null distribution.
    """

    real: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    gap_changes: dict[str, list[float]] = field(default_factory=dict)
    null_band: list[float] = field(default_factory=list)


def collect_scores(all_trials: list[TrialResult], site: InjectionSite) -> ScoreTable:
    """Compute every family's real and control binding scores at one site.

    Filters to ROLE-probe trials at `site`, groups by family, and computes
    family_binding_score wherever ROLE_PUSH trials exist plus one
    control_binding_score per control edit present. The entity token is
    recovered from pair_id ("entity->counterpart", see ConceptPair.pair_id)
    since TrialResult does not carry the AnswerSet. Incomplete cells inside
    any attempted score raise (via _cell_logits) rather than being skipped.
    """
    by_family: dict[str, list[TrialResult]] = {}
    for t in all_trials:
        if t.probe_kind is ProbeKind.ROLE and t.injection_site is site:
            by_family.setdefault(t.family_id, []).append(t)

    table = ScoreTable()
    for trials in by_family.values():
        first = trials[0]
        entity_token = first.pair_id.split("->", 1)[0]
        edits_present = {t.edit_type for t in trials}
        if EditType.ROLE_PUSH in edits_present:
            group_key = (first.construction.value, first.pair_id)
            table.real.setdefault(group_key, []).append(
                family_binding_score(trials, entity_token)
            )
            for sign, value in family_gap_changes(trials, entity_token).items():
                table.gap_changes.setdefault(sign.value, []).append(value)
        for control in CONTROL_EDIT_TYPES:
            if control in edits_present:
                table.null_band.append(control_binding_score(trials, entity_token, control))
    return table
