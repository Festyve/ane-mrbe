"""Binding score: position-averaged, both-sign log-odds crossover per ItemFamily.

Pure numpy. Probabilities arrive as TrialResult records; this module reads
`answer_probs[entity]` (the pushed entity, NOT the identity-swap counterpart)
and converts to log-odds without renormalizing.

Log-odds rather than raw probability because a role-blind uniform push shifts
the entity's log-odds by the same increment in both role conditions regardless
of where they started, so floor/ceiling effects cannot fake a role interaction.

With `L(role, edit, sign)` the logit P(entity) at the ROLE probe averaged over
the two surface positions, the per-sign gap change and family score are

    dG(s) = [L(agent, push_s) - L(patient, push_s)]
          - [L(agent, no_edit) - L(patient, no_edit)]
    BS_i  = -(dG(toward_agent) + dG(toward_patient)) / 2

A bag workspace leaves dG(s) ~ 0 for both signs; a binding workspace shrinks
the natural gap from opposite sides (the crossover), making both negative and
BS_i positive. Control edits plug into the same statistic in place of ROLE_PUSH
and form the null band. A within-family agent/patient swap negates BS_i, so the
sign-flip permutation test in analysis.stats is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from jspace_binding.types import (
    ConceptPair,
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

# The three constructions testing agent/patient proper, which the "meaningful in
# >= 2 of 3 constructions" criterion quantifies over. DATIVE is reported
# separately: it contrasts giver against recipient, and a recipient is a goal
# rather than a patient, so it measures a different thematic relation.
AGENT_PATIENT_CONSTRUCTIONS: tuple[str, ...] = (
    "active_passive",
    "cleft",
    "relative_clause",
)

_EPS = 1e-9  # probability clamp so a hard 0/1 read cannot produce +/- inf


def logit(p: float) -> float:
    """log(p / (1-p)) with clamping to (_EPS, 1-_EPS)."""
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def position_average(values: dict[tuple[Role, Position], float]) -> dict[Role, float]:
    """Collapse the position nuisance axis: L(role) = mean over FIRST/SECOND.

    Any surface-order bias contributes equally to both roles and cancels in the
    gap. A role present with only one position would reintroduce that confound,
    so it raises instead.
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
    trials: list[TrialResult],
    entity_token: str,
    treatment: EditType,
    probe_kind: ProbeKind = ProbeKind.ROLE,
) -> dict[_CellKey, float]:
    """Logit P(entity) per (role, position, edit, sign) cell at one probe.

    Requires exactly one trial per cell — 12 in total. Missing or duplicate
    cells raise ValueError naming the cell, never a silently wrong score.
    """
    cells: dict[_CellKey, float] = {}
    for t in trials:
        if t.probe_kind is not probe_kind:
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
            f"cannot compute binding score vs {treatment.value!r}: missing "
            f"{probe_kind.value}-probe cells " + ", ".join(missing)
        )
    return cells


def _cell_name(key: _CellKey) -> str:
    role, pos, edit, sign = key
    sign_txt = "" if sign is None else f", {sign.value}"
    return f"({role.value}, {pos.value}, {edit.value}{sign_txt})"


def family_gap_changes(
    trials: list[TrialResult],
    entity_token: str,
    treatment: EditType = EditType.ROLE_PUSH,
    probe_kind: ProbeKind = ProbeKind.ROLE,
) -> dict[PushSign, float]:
    """dG(sign) for both signs of one family.

    Binding predicts both negative; an asymmetry (promotion works, demotion
    does not) shows up here rather than being averaged away. Raw, in the
    probe's own orientation — PROBE_ORIENTATION is applied by the score
    functions, so this always describes what the probe literally measured.
    """
    cells = _cell_logits(trials, entity_token, treatment, probe_kind)
    baseline = position_average(
        {
            (role, pos): cells[(role, pos, EditType.NO_EDIT, None)]
            for role in Role
            for pos in Position
        }
    )
    natural_gap = baseline[Role.AGENT] - baseline[Role.PATIENT]
    gaps: dict[PushSign, float] = {}
    for sign in PushSign:
        pushed = position_average(
            {
                (role, pos): cells[(role, pos, treatment, sign)]
                for role in Role
                for pos in Position
            }
        )
        gaps[sign] = float((pushed[Role.AGENT] - pushed[Role.PATIENT]) - natural_gap)
    return gaps


def _crossover_score(gaps: dict[PushSign, float]) -> float:
    return -(gaps[PushSign.TOWARD_AGENT] + gaps[PushSign.TOWARD_PATIENT]) / 2.0


# Which way "more probable entity" points relative to the AGENT role. The
# RECIPIENT probe inverts, so negating keeps "positive means binding" shared
# between the two readouts. Applied once, in the score functions.
PROBE_ORIENTATION: dict[ProbeKind, float] = {
    ProbeKind.ROLE: 1.0,
    ProbeKind.RECIPIENT: -1.0,
}


def family_binding_score(
    trials: list[TrialResult],
    entity_token: str,
    probe_kind: ProbeKind = ProbeKind.ROLE,
) -> float:
    """Binding score of one family from its trials at one site and probe.

    Positive under binding, ~0 under a bag workspace. Requires all 12 cells.
    Site filtering is the caller's job (collect_scores).
    """
    gaps = family_gap_changes(trials, entity_token, EditType.ROLE_PUSH, probe_kind)
    return PROBE_ORIENTATION[probe_kind] * _crossover_score(gaps)


def control_binding_score(
    trials: list[TrialResult],
    entity_token: str,
    edit_type: EditType,
    probe_kind: ProbeKind = ProbeKind.ROLE,
) -> float:
    """Same crossover statistic with ROLE_PUSH replaced by a control edit.

    The control carries no role information for this sentence, so its score
    should sit near 0; their empirical distribution is the null band.
    """
    if edit_type not in CONTROL_EDIT_TYPES:
        allowed = ", ".join(e.value for e in CONTROL_EDIT_TYPES)
        raise ValueError(
            f"control_binding_score: {edit_type.value!r} is not a control edit ({allowed})"
        )
    gaps = family_gap_changes(trials, entity_token, edit_type, probe_kind)
    return PROBE_ORIENTATION[probe_kind] * _crossover_score(gaps)


@dataclass
class ScoreTable:
    """Family-level scores grouped for stats and the forest plot.

    `real` is keyed by (construction, pair_id), keeping constructions separate
    so a single-construction effect cannot masquerade as general binding.
    `null_band` pools every control score into one null distribution;
    `null_by_edit` keeps them split so the verdict can check each control —
    the random direction in particular — individually against the band.
    """

    real: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    gap_changes: dict[str, list[float]] = field(default_factory=dict)
    null_band: list[float] = field(default_factory=list)
    null_by_edit: dict[str, list[float]] = field(default_factory=dict)


def collect_scores(
    all_trials: list[TrialResult],
    site: InjectionSite,
    probe_kind: ProbeKind = ProbeKind.ROLE,
) -> ScoreTable:
    """Compute every family's real and control binding scores at one site.

    The entity token is recovered from pair_id, since TrialResult does not
    carry the AnswerSet. Incomplete cells raise rather than being skipped.
    """
    orientation = PROBE_ORIENTATION[probe_kind]
    by_family: dict[str, list[TrialResult]] = {}
    for t in all_trials:
        if t.probe_kind is probe_kind and t.injection_site is site:
            by_family.setdefault(t.family_id, []).append(t)

    table = ScoreTable()
    for trials in by_family.values():
        first = trials[0]
        entity_token = ConceptPair.entity_of(first.pair_id)
        edits_present = {t.edit_type for t in trials}
        if EditType.ROLE_PUSH in edits_present:
            gaps = family_gap_changes(trials, entity_token, EditType.ROLE_PUSH, probe_kind)
            group_key = (first.construction.value, first.pair_id)
            table.real.setdefault(group_key, []).append(orientation * _crossover_score(gaps))
            for sign, value in gaps.items():
                table.gap_changes.setdefault(sign.value, []).append(value)
        for control in CONTROL_EDIT_TYPES:
            if control in edits_present:
                score = control_binding_score(trials, entity_token, control, probe_kind)
                table.null_band.append(score)
                table.null_by_edit.setdefault(control.value, []).append(score)
    return table
