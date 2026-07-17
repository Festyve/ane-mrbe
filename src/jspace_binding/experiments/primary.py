"""Primary experiment: the full condition sweep and the analysis behind the paper numbers.

run_primary is written strictly against the WorkspaceModel protocol, so the
identical sweep drives the GPU-free DummyModel (validating the analysis against
known ground truth) and, later, the real Qwen + J-lens backend.

Probe assignment per edit type (proposal, Experimental Setup §2):
- direction pushes (ROLE_PUSH and its three controls): ROLE probe, both signs;
- NO_EDIT: ROLE probe (the DiD baseline) plus NEUTRAL probe (the strength
  check's reference point);
- IDENTITY_SWAP: NEUTRAL probe only — the intervention-strength control never
  enters the binding score.

analyze() turns the trial list into per-(construction, pair) binding scores
with bootstrap CIs, permutation p-values (Holm-corrected across groups),
Cohen's d, the control null band, the per-sign gap-change breakdown, the
neutral-probe strength check, and both figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jspace_binding.analysis import stats
from jspace_binding.analysis.binding_score import collect_scores
from jspace_binding.analysis.plots import forest_plot, per_condition_plot
from jspace_binding.config import Config
from jspace_binding.interventions.edits import plan_edit
from jspace_binding.model.base import WorkspaceModel
from jspace_binding.types import (
    DIRECTION_PUSH_EDIT_TYPES,
    EditType,
    InjectionSite,
    ItemFamily,
    Position,
    ProbeKind,
    PushSign,
    Role,
    TrialResult,
)


def run_primary(
    config: Config, model: WorkspaceModel, families: list[ItemFamily]
) -> list[TrialResult]:
    """Sweep family x (role x position) cell x (edit type x sign) x injection site.

    Trials are written to config.paths.results / "trials.jsonl" and returned
    in sweep order.
    """
    trials: list[TrialResult] = []
    for family_index, family in enumerate(families):
        if family.answer_set is None:
            raise ValueError(f"family {family.family_id!r} has no answer_set")
        for edit_type in config.experiment.edit_types:
            for sign, probe_kinds in _conditions(config, edit_type):
                edit = plan_edit(
                    family,
                    edit_type,
                    sign=sign,
                    non_participant_candidates=config.stimuli.non_participant_entities,
                    alpha=config.model.alpha,
                    coefficient=config.model.push_coefficient,
                    # Offset per family so RANDOM_DIRECTION draws a fresh,
                    # reproducible vector per family; the two signs of one
                    # family push the SAME vector in opposite directions,
                    # mirroring how r_entity is used.
                    seed=config.experiment.seed + family_index,
                )
                for role in Role:
                    for position in Position:
                        stimulus = family.cell(role, position)
                        for site in config.experiment.injection_sites:
                            for probe_kind in probe_kinds:
                                probe = (
                                    family.role_probe
                                    if probe_kind is ProbeKind.ROLE
                                    else family.neutral_probe
                                )
                                answer_probs = model.answer_distribution(
                                    stimulus.sentence,
                                    probe,
                                    edit,
                                    site,
                                    family.answer_set.tokens,
                                )
                                trials.append(
                                    TrialResult(
                                        family_id=family.family_id,
                                        pair_id=family.concept_pair.pair_id,
                                        construction=family.construction,
                                        role=role,
                                        position=position,
                                        edit_type=edit_type,
                                        probe_kind=probe_kind,
                                        injection_site=site,
                                        answer_probs={
                                            token: float(p)
                                            for token, p in answer_probs.items()
                                        },
                                        push_sign=sign,
                                    )
                                )
    _write_trials(trials, Path(config.paths.results) / "trials.jsonl")
    return trials


def _conditions(
    config: Config, edit_type: EditType
) -> list[tuple[PushSign | None, tuple[ProbeKind, ...]]]:
    """(sign, probe kinds) conditions for one edit type."""
    if edit_type in DIRECTION_PUSH_EDIT_TYPES:
        return [(sign, (ProbeKind.ROLE,)) for sign in config.experiment.push_signs]
    if edit_type is EditType.NO_EDIT:
        return [(None, (ProbeKind.ROLE, ProbeKind.NEUTRAL))]
    if edit_type is EditType.IDENTITY_SWAP:
        return [(None, (ProbeKind.NEUTRAL,))]
    raise ValueError(f"Unhandled edit type in sweep: {edit_type!r}")


def analyze(config: Config, trials: list[TrialResult]) -> dict[str, object]:
    """Score, test, and plot the primary (final-token) analysis.

    Permutation p-values are Holm-corrected across the (construction x pair)
    groups; the pooled row is the omnibus and stays uncorrected. Everything is
    cast to plain Python types, so the summary is json.dumps-able as returned.
    """
    if not trials:
        raise ValueError("analyze: no trials to analyze")
    site = InjectionSite.FINAL_TOKEN  # primary site; ENTITY_TOKEN is the secondary analysis
    table = collect_scores(trials, site)
    if not table.real:
        raise ValueError("analyze: no ROLE_PUSH ROLE-probe trials at the final-token site")
    band_lo, band_hi = stats.null_band(table.null_band, ci_level=config.analysis.ci_level)

    group_stats = {key: _stats_block(scores, config) for key, scores in table.real.items()}
    significant = stats.holm_bonferroni(
        {key: block["p_perm"] for key, block in group_stats.items()},
        alpha=config.analysis.alpha_level,
    )
    pooled = _stats_block([score for scores in table.real.values() for score in scores], config)

    figures_dir = Path(config.paths.figures)
    forest_path = figures_dir / "forest.png"
    per_condition_path = figures_dir / "per_condition.png"
    forest_plot(table, group_stats, (band_lo, band_hi), forest_path)
    # The per-condition figure reads one pair's entity token; the sweep's first
    # trial makes the choice deterministic and guaranteed present in the data.
    example_entity = trials[0].pair_id.split("->", 1)[0]
    per_condition_plot(trials, example_entity, per_condition_path)

    return {
        "site": site.value,
        "ci_level": float(config.analysis.ci_level),
        "n_trials": len(trials),
        "null_band": [float(band_lo), float(band_hi)],
        "n_null_scores": len(table.null_band),
        "pooled": pooled,
        "gap_change_by_sign": {
            sign: {"n": len(values), "mean": float(np.mean(values))}
            for sign, values in table.gap_changes.items()
        },
        "neutral_strength_check": _neutral_strength_check(trials, site),
        "groups": {
            f"{construction}|{pair_id}": {
                "construction": construction,
                "pair_id": pair_id,
                **group_stats[(construction, pair_id)],
                "significant_holm": bool(significant[(construction, pair_id)]),
            }
            for construction, pair_id in group_stats
        },
        "figures": {"forest": str(forest_path), "per_condition": str(per_condition_path)},
    }


def _neutral_strength_check(trials: list[TrialResult], site: InjectionSite) -> dict[str, object]:
    """The load-bearing intervention-strength control (proposal, §3 Controls).

    Compares P(counterpart) and P(entity) at the NEUTRAL probe under
    IDENTITY_SWAP against NO_EDIT. Passing (counterpart rises, entity falls)
    licenses interpreting a null binding score as evidence about binding
    rather than a dead intervention; a failure makes any null uninterpretable
    and analyze() surfaces that verdict instead of hiding it.
    """
    sums: dict[tuple[EditType, str], list[float]] = {}
    for t in trials:
        if t.probe_kind is not ProbeKind.NEUTRAL or t.injection_site is not site:
            continue
        if t.edit_type not in (EditType.IDENTITY_SWAP, EditType.NO_EDIT):
            continue
        entity, counterpart = t.pair_id.split("->", 1)
        sums.setdefault((t.edit_type, "entity"), []).append(t.answer_probs[entity])
        sums.setdefault((t.edit_type, "counterpart"), []).append(t.answer_probs[counterpart])

    needed = [(e, tok) for e in (EditType.IDENTITY_SWAP, EditType.NO_EDIT)
              for tok in ("entity", "counterpart")]
    if any(key not in sums for key in needed):
        return {
            "available": False,
            "note": "needs identity_swap and no_edit NEUTRAL-probe trials at this site",
        }
    mean = {key: float(np.mean(values)) for key, values in sums.items()}
    counterpart_shift = (
        mean[(EditType.IDENTITY_SWAP, "counterpart")] - mean[(EditType.NO_EDIT, "counterpart")]
    )
    entity_shift = mean[(EditType.IDENTITY_SWAP, "entity")] - mean[(EditType.NO_EDIT, "entity")]
    return {
        "available": True,
        "p_counterpart_no_edit": mean[(EditType.NO_EDIT, "counterpart")],
        "p_counterpart_swap": mean[(EditType.IDENTITY_SWAP, "counterpart")],
        "p_entity_no_edit": mean[(EditType.NO_EDIT, "entity")],
        "p_entity_swap": mean[(EditType.IDENTITY_SWAP, "entity")],
        "counterpart_shift": counterpart_shift,
        "entity_shift": entity_shift,
        "passes": bool(counterpart_shift > 0.0 and entity_shift < 0.0),
    }


def _stats_block(scores: list[float], config: Config) -> dict[str, float | int]:
    """Mean, bootstrap CI, permutation p, and effect size for one score list."""
    ci_lo, ci_hi = stats.bootstrap_ci(
        scores,
        n_resamples=config.analysis.n_bootstrap,
        ci_level=config.analysis.ci_level,
        seed=config.experiment.seed,
    )
    p_perm = stats.permutation_pvalue(
        scores, n_permutations=config.analysis.n_permutation, seed=config.experiment.seed
    )
    return {
        "n": len(scores),
        "mean": float(np.mean(scores)),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_perm": float(p_perm),
        "cohens_d": float(stats.cohens_d(scores)),
    }


def _write_trials(trials: list[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in trials:
            record = {
                "family_id": t.family_id,
                "pair_id": t.pair_id,
                "construction": t.construction.value,
                "role": t.role.value,
                "position": t.position.value,
                "edit_type": t.edit_type.value,
                "probe_kind": t.probe_kind.value,
                "injection_site": t.injection_site.value,
                "answer_probs": t.answer_probs,
                "push_sign": None if t.push_sign is None else t.push_sign.value,
            }
            f.write(json.dumps(record) + "\n")
