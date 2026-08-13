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
from jspace_binding.analysis.binding_score import (
    AGENT_PATIENT_CONSTRUCTIONS,
    CONTROL_EDIT_TYPES,
    ScoreTable,
    _cell_logits,
    collect_scores,
)
from jspace_binding.analysis.plots import forest_plot, per_condition_plot
from jspace_binding.config import Config
from jspace_binding.experiments.progress import track
from jspace_binding.interventions.edits import plan_edit
from jspace_binding.model.base import WorkspaceModel
from jspace_binding.types import (
    DIRECTION_PUSH_EDIT_TYPES,
    ConceptPair,
    EditType,
    InjectionSite,
    ItemFamily,
    Position,
    ProbeKind,
    PushSign,
    Role,
    TrialResult,
)


def validate_config(config: Config) -> None:
    """Reject configs the primary statistic cannot support, BEFORE any model
    call — a bad sweep config must fail in milliseconds, not after GPU hours.

    - RQ2 ablation edit types never belong in the primary sweep (they have
      their own runner, experiments.rq2_ablation).
    - The crossover binding score is defined over BOTH push signs; a
      single-sign config would only crash in analyze() after the sweep spent
      its compute (analysis.binding_score._cell_logits requires both).
    """
    rq2_only = [
        e.value
        for e in config.experiment.edit_types
        if e in (EditType.ABLATE_JSPACE, EditType.ABLATE_RANDOM_SUBSPACE)
    ]
    if rq2_only:
        raise ValueError(
            f"edit_types {rq2_only} are RQ2 ablations, not primary-sweep conditions; "
            "run scripts/run_rq2.py instead"
        )
    has_push = any(e in DIRECTION_PUSH_EDIT_TYPES for e in config.experiment.edit_types)
    if has_push and set(config.experiment.push_signs) != set(PushSign):
        got = [s.value for s in config.experiment.push_signs]
        raise ValueError(
            f"the crossover binding score needs both push signs, got {got}; "
            "set experiment.push_signs: [toward_agent, toward_patient]"
        )


def run_primary(
    config: Config,
    model: WorkspaceModel,
    families: list[ItemFamily],
    site: InjectionSite | None = None,
) -> list[TrialResult]:
    """Sweep family x (role x position) cell x (edit type x sign) x injection site.

    `site` restricts the sweep to one injection site; None sweeps every site in
    experiment.injection_sites. Each site is a full sweep, so naming one halves
    GPU cost — analyze() reads a single site anyway, so sweeping both only pays
    off once something reads the second one back.

    Trials are written to config.paths.results / "trials.jsonl" and returned
    in sweep order.
    """
    validate_config(config)
    sites = (site,) if site is not None else tuple(config.experiment.injection_sites)
    trials: list[TrialResult] = []
    for family_index, family in enumerate(
        track(families, "primary", total=len(families))
    ):
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
                        for trial_site in sites:
                            for probe_kind in probe_kinds:
                                probe = _probe_text(family, probe_kind)
                                if not probe:
                                    # RECIPIENT is dative-only; the other three
                                    # constructions carry no recipient probe.
                                    continue
                                answer_probs = model.answer_distribution(
                                    stimulus.sentence,
                                    probe,
                                    edit,
                                    trial_site,
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
                                        injection_site=trial_site,
                                        answer_probs={
                                            token: float(p)
                                            for token, p in answer_probs.items()
                                        },
                                        push_sign=sign,
                                    )
                                )
    _write_trials(trials, Path(config.paths.results) / "trials.jsonl")
    return trials


def _probe_text(family: ItemFamily, probe_kind: ProbeKind) -> str:
    """The probe string for one kind, or "" when this family has none."""
    if probe_kind is ProbeKind.ROLE:
        return family.role_probe
    if probe_kind is ProbeKind.NEUTRAL:
        return family.neutral_probe
    if probe_kind is ProbeKind.RECIPIENT:
        return family.recipient_probe
    raise ValueError(f"Unhandled probe kind in sweep: {probe_kind!r}")


def _conditions(
    config: Config, edit_type: EditType
) -> list[tuple[PushSign | None, tuple[ProbeKind, ...]]]:
    """(sign, probe kinds) conditions for one edit type.

    RECIPIENT rides along with ROLE on the pushes and the no-edit baseline so
    the dative gets a complete 12-cell score (and its own null band) from the
    recipient side too. Non-dative families skip it in the sweep loop.
    """
    if edit_type in DIRECTION_PUSH_EDIT_TYPES:
        return [
            (sign, (ProbeKind.ROLE, ProbeKind.RECIPIENT))
            for sign in config.experiment.push_signs
        ]
    if edit_type is EditType.NO_EDIT:
        return [(None, (ProbeKind.ROLE, ProbeKind.NEUTRAL, ProbeKind.RECIPIENT))]
    if edit_type is EditType.IDENTITY_SWAP:
        return [(None, (ProbeKind.NEUTRAL,))]
    raise ValueError(f"Unhandled edit type in sweep: {edit_type!r}")


def analyze(
    config: Config, trials: list[TrialResult], site: InjectionSite | None = None
) -> dict[str, object]:
    """Score, test, and plot the primary analysis at one injection site.

    Defaults to FINAL_TOKEN, the primary site; ENTITY_TOKEN is the secondary
    analysis. Only one site is ever scored, so a sweep that covered both still
    yields a single-site summary — pass `site` to pick which.

    Permutation p-values are Holm-corrected across the (construction x pair)
    groups; the pooled row is the omnibus and stays uncorrected. Everything is
    cast to plain Python types, so the summary is json.dumps-able as returned.
    """
    if not trials:
        raise ValueError("analyze: no trials to analyze")
    if site is None:
        site = InjectionSite.FINAL_TOKEN
    table = collect_scores(trials, site)
    if not table.real:
        raise ValueError(
            f"analyze: no ROLE_PUSH ROLE-probe trials at the {site.value} site; "
            "the sweep did not cover it (see run_primary --site)"
        )
    band_lo, band_hi = stats.null_band(table.null_band, ci_level=config.analysis.ci_level)

    group_stats = {key: _stats_block(scores, config) for key, scores in table.real.items()}
    significant = stats.holm_bonferroni(
        {key: block["p_perm"] for key, block in group_stats.items()},
        alpha=config.analysis.alpha_level,
    )
    pooled = _stats_block([score for scores in table.real.values() for score in scores], config)

    # Per-construction blocks (pooled over pairs): the positive-result
    # criterion quantifies over constructions, not (construction x pair)
    # slices (proposal, Ideal Results #3). Holm-corrected within this family.
    by_construction: dict[str, list[float]] = {}
    for (construction, _), scores in table.real.items():
        by_construction.setdefault(construction, []).extend(scores)
    construction_stats = {
        construction: _stats_block(scores, config)
        for construction, scores in by_construction.items()
    }
    construction_significant = stats.holm_bonferroni(
        {c: block["p_perm"] for c, block in construction_stats.items()},
        alpha=config.analysis.alpha_level,
    )
    neutral_check = _neutral_strength_check(trials, site)
    push_check = _push_strength_check(trials, site, config)
    verdict = _verdict(
        config,
        pooled,
        construction_stats,
        construction_significant,
        table,
        (band_lo, band_hi),
        push_check,
        neutral_check,
    )

    figures_dir = Path(config.paths.figures)
    forest_path = figures_dir / "forest.png"
    per_condition_path = figures_dir / "per_condition.png"
    forest_plot(table, group_stats, (band_lo, band_hi), forest_path)
    # The per-condition figure reads one pair's entity token; the sweep's first
    # trial makes the choice deterministic and guaranteed present in the data.
    example_entity = ConceptPair.entity_of(trials[0].pair_id)
    per_condition_plot(trials, example_entity, per_condition_path, site=site)

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
        "push_strength_check": push_check,
        "neutral_strength_check": neutral_check,
        "constructions": {
            construction: {
                **construction_stats[construction],
                "significant_holm": bool(construction_significant[construction]),
                "agent_patient_family": construction in AGENT_PATIENT_CONSTRUCTIONS,
            }
            for construction in construction_stats
        },
        "groups": {
            f"{construction}|{pair_id}": {
                "construction": construction,
                "pair_id": pair_id,
                **group_stats[(construction, pair_id)],
                "significant_holm": bool(significant[(construction, pair_id)]),
            }
            for construction, pair_id in group_stats
        },
        "verdict": verdict,
        "figures": {"forest": str(forest_path), "per_condition": str(per_condition_path)},
    }


# P(counterpart) must rise by at least this much for the swap to count as
# having landed. Identical to experiments.calibrate.calibrate_identity_alpha's
# `min_prob_shift`, deliberately: calibration and the in-run check are the same
# question asked twice, and they MUST agree.
#
# They did not. This check tested only SIGNS, and on Qwen3.6-27B it passed on
# counterpart 0.0027 -> 0.0039 (+0.0012) with the entity flat at 0.185 -> 0.181
# -- nothing moved -- while calibration on the same model reported an
# intervention-strength FAILURE at the same threshold it applies here. The
# paper's one positive claim, the concept-vs-role addressability dissociation,
# rested on that pass.
_MIN_COUNTERPART_SHIFT = 0.05


def _neutral_strength_check(trials: list[TrialResult], site: InjectionSite) -> dict[str, object]:
    """The load-bearing intervention-strength control (proposal, §3 Controls).

    Compares P(counterpart) and P(entity) at the NEUTRAL probe under
    IDENTITY_SWAP against NO_EDIT. Passing licenses interpreting a null binding
    score as evidence about binding rather than a dead intervention; a failure
    makes any null uninterpretable and analyze() surfaces that verdict instead
    of hiding it.

    Passing requires the counterpart to rise by a MAGNITUDE
    (`_MIN_COUNTERPART_SHIFT`), not merely to rise. A swap that nudges the
    counterpart from 0.3% to 0.4% has not installed anything, and licensing a
    null on it asserts exactly what the check exists to rule out. This is the
    same failure the project has hit repeatedly: a threshold comparing two
    quantities without first asking whether either is distinguishable from
    nothing.
    """
    sums: dict[tuple[EditType, str], list[float]] = {}
    for t in trials:
        if t.probe_kind is not ProbeKind.NEUTRAL or t.injection_site is not site:
            continue
        if t.edit_type not in (EditType.IDENTITY_SWAP, EditType.NO_EDIT):
            continue
        entity, counterpart = ConceptPair.split_pair_id(t.pair_id)
        missing = [tok for tok in (entity, counterpart) if tok not in t.answer_probs]
        if missing:
            raise ValueError(
                f"neutral strength check: answer tokens {missing} missing from "
                f"answer_probs of a {t.edit_type.value} trial in family {t.family_id!r}"
            )
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
        "min_counterpart_shift": _MIN_COUNTERPART_SHIFT,
        "passes": bool(
            counterpart_shift >= _MIN_COUNTERPART_SHIFT and entity_shift < 0.0
        ),
    }


# Minimum mean |logit| displacement of the readout under ROLE_PUSH for the
# arm to count as live. Identical to experiments.calibrate's `min_logit_shift`,
# deliberately and for the same reason _MIN_COUNTERPART_SHIFT matches
# calibrate_identity_alpha: calibration and the in-run check are the same
# question asked twice and MUST agree. Qwen's diff-means push moved the readout
# ~0.01 log-odds non-monotonically across a 16x range (RESULTS.md §7) and is
# the case this must exclude; 0.5 excludes it by a wide margin.
_MIN_PUSH_DISPLACEMENT = 0.5


def _readout_displacement(
    trials: list[TrialResult], entity_token: str, treatment: EditType
) -> float:
    """Mean |logit P(entity) under `treatment` - under NO_EDIT| over all cells.

    A LEVEL statistic, not a gap contrast, and that choice is load-bearing.

    The obvious candidate was the "steering mirror", the difference of the two
    signed gap changes, chosen because the binding score is their sum and the
    two are orthogonal contrasts. It is wrong. Both gap changes are computed
    against the SAME natural gap, so a push whose effect is uniform across
    roles cancels entirely -- and that is exactly what a bag workspace does.
    In DummyModel the mirror is identically zero in BOTH modes (binding:
    dG = -2.2 under either sign; bag: a uniform sign*0.8 shift that the gap
    subtracts away), so gating on it fails the ground-truth validation the
    whole pipeline rests on, in the mode that is supposed to be a clean
    negative.

    Displacement has the property the gate actually needs. It is large
    whenever the push moves the readout at all, in binding mode (~2.2) and bag
    mode (0.8) alike, while the binding score is ~0 in bag mode -- so a null
    binding result cannot fail its own strength check. That is the
    non-circularity requirement, and it is met without requiring the null to
    also be role-structured.
    """
    cells = _cell_logits(trials, entity_token, treatment, ProbeKind.ROLE)
    baseline = {
        (role, pos): cells[(role, pos, EditType.NO_EDIT, None)]
        for role in Role
        for pos in Position
    }
    shifts = [
        abs(value - baseline[(role, pos)])
        for (role, pos, edit, _sign), value in cells.items()
        if edit is treatment
    ]
    return float(np.mean(shifts)) if shifts else 0.0


def _push_strength_check(
    trials: list[TrialResult], site: InjectionSite, config: Config
) -> dict[str, object]:
    """Intervention-strength control for the arm the binding score is built on.

    The binding score is a ROLE_PUSH quantity. `_neutral_strength_check` tests
    IDENTITY_SWAP, a different edit serving the separate (and retracted)
    addressability claim, whose alpha failed to calibrate on both models
    tested. Gating a push result on a swap control conflates two independent
    claims and makes every push null uninterpretable by construction.

    Passing requires BOTH:
      1. mean readout displacement >= _MIN_PUSH_DISPLACEMENT -- an absolute
         floor, because "significantly greater than a control" is satisfiable
         by an arbitrarily tiny effect given enough families; and
      2. a bootstrap CI on the paired difference against the STRONGEST control
         edit that excludes zero -- so the movement is specific to the fitted
         role direction rather than what any push of that norm would do.

    Paired per family, so (1) and (2) describe the same families.
    """
    by_family: dict[str, list[TrialResult]] = {}
    for t in trials:
        if t.probe_kind is ProbeKind.ROLE and t.injection_site is site:
            by_family.setdefault(t.family_id, []).append(t)

    role: list[float] = []
    controls: dict[str, list[float]] = {}
    for family_trials in by_family.values():
        edits = {t.edit_type for t in family_trials}
        if EditType.ROLE_PUSH not in edits:
            continue
        entity = ConceptPair.entity_of(family_trials[0].pair_id)
        role.append(_readout_displacement(family_trials, entity, EditType.ROLE_PUSH))
        for control in CONTROL_EDIT_TYPES:
            if control in edits:
                controls.setdefault(control.value, []).append(
                    _readout_displacement(family_trials, entity, control)
                )

    if not role:
        return {
            "available": False,
            "note": f"no ROLE_PUSH ROLE-probe trials at {site.value}",
        }

    role_mean = float(np.mean(role))
    control_means = {edit: float(np.mean(values)) for edit, values in controls.items()}
    # Strongest control = most conservative comparison available. Which control
    # that is varies by model and must not be hard-coded: on the Gemma LRE run
    # it is null_non_participant (pushing an ABSENT entity's direction), not
    # random_direction, and comparing against random alone would overstate the
    # margin several-fold.
    strongest = max(control_means, key=lambda e: control_means[e]) if control_means else None

    ci_lo = ci_hi = None
    beats_control = False
    if strongest is not None and len(controls[strongest]) == len(role):
        paired = [r - c for r, c in zip(role, controls[strongest], strict=True)]
        ci_lo, ci_hi = stats.bootstrap_ci(
            paired,
            n_resamples=config.analysis.n_bootstrap,
            ci_level=config.analysis.ci_level,
            seed=config.experiment.seed,
        )
        beats_control = bool(ci_lo > 0.0)

    clears_floor = role_mean >= _MIN_PUSH_DISPLACEMENT
    return {
        "available": True,
        "role_push_displacement": role_mean,
        "control_displacement": control_means,
        "strongest_control": strongest,
        "vs_control_ci": [ci_lo, ci_hi] if ci_lo is not None else None,
        "min_push_displacement": _MIN_PUSH_DISPLACEMENT,
        "clears_floor": bool(clears_floor),
        "beats_strongest_control": beats_control,
        "passes": bool(clears_floor and beats_control),
    }


_D_MEANINGFUL = 0.5  # a-priori effect-size threshold (proposal, §6)
_D_CI_TIGHT = 1.0  # max CI width on d for a null to count as "clean" not "underpowered"


def _stats_block(scores: list[float], config: Config) -> dict[str, float | int]:
    """Mean, bootstrap CI, permutation p, and effect size (with its own
    bootstrap CI, per proposal §6) for one score list."""
    ci_lo, ci_hi = stats.bootstrap_ci(
        scores,
        n_resamples=config.analysis.n_bootstrap,
        ci_level=config.analysis.ci_level,
        seed=config.experiment.seed,
    )
    p_perm = stats.permutation_pvalue(
        scores, n_permutations=config.analysis.n_permutation, seed=config.experiment.seed
    )
    d_ci_lo, d_ci_hi = stats.cohens_d_ci(
        scores,
        n_resamples=config.analysis.n_bootstrap,
        ci_level=config.analysis.ci_level,
        seed=config.experiment.seed,
    )
    return {
        "n": len(scores),
        "mean": float(np.mean(scores)),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_perm": float(p_perm),
        "cohens_d": float(stats.cohens_d(scores)),
        "cohens_d_ci_lo": float(d_ci_lo),
        "cohens_d_ci_hi": float(d_ci_hi),
    }


def _verdict(
    config: Config,
    pooled: dict[str, float | int],
    construction_stats: dict[str, dict[str, float | int]],
    construction_significant: dict[str, bool],
    table: ScoreTable,
    band: tuple[float, float],
    push_check: dict[str, object],
    neutral_check: dict[str, object],
) -> dict[str, object]:
    """Classify the outcome per the proposal's Benchmarks / Ideal Results.

    Outcomes:
    - uninterpretable_strength_failure: the ROLE_PUSH arm did not move the
      readout, so no binding conclusion is licensed either way.
    - indistinguishable_from_controls: pooled-significant, but the pooled mean
      lies INSIDE the null band the control edits define. Significance and
      effect size both measure how CONSISTENTLY the push moves the score, not
      whether the movement has anything to do with role structure; a
      strength-matched push of a direction belonging to an entity absent from
      the sentence moves it just as far. Checked before effect size, because
      "how big" is only meaningful once "distinguishable from nothing" is
      settled.
    - positive_binding: pooled significant with d >= 0.5, the pooled mean above
      the null band, the effect holding (Holm-significant AND d >= 0.5) in
      >= 2 of the 3 agent/patient constructions, and the random-direction
      control inside the band.
    - significant_but_tiny: clears the null band but is too weak to support the
      monitoring agenda in practice (d < 0.5) — a different story, told
      separately.
    - suggestive_not_conclusive: pooled-significant but the per-construction
      criterion is unmet (e.g. carried by one construction, or underpowered
      slices).
    - clean_negative: not significant AND the CI on d is tight — evidence FOR
      the bag-of-concepts answer, reported as a positive finding.
    - inconclusive_underpowered: not significant with a wide CI on d —
      stimulus expansion is the remedy, not a conclusion.
    """
    # Gate on the arm under test. The binding score is built from ROLE_PUSH
    # trials, so ROLE_PUSH is the intervention whose strength licenses (or
    # refuses) a conclusion about it. IDENTITY_SWAP is still computed and
    # reported, but it gates the addressability claim only -- letting it gate
    # here made every push result on both models tested uninterpretable by
    # construction, including runs whose push calibrated cleanly.
    strength_passes = bool(push_check.get("available")) and bool(push_check.get("passes"))
    band_lo, band_hi = band
    random_scores = table.null_by_edit.get("random_direction", [])
    random_mean = float(np.mean(random_scores)) if random_scores else None
    random_in_band = bool(
        random_scores and band_lo <= random_mean <= band_hi  # type: ignore[operator]
    )
    meaningful = [
        construction
        for construction in AGENT_PATIENT_CONSTRUCTIONS
        if construction in construction_stats
        and construction_significant[construction]
        and construction_stats[construction]["cohens_d"] >= _D_MEANINGFUL
    ]
    pooled_significant = pooled["p_perm"] < config.analysis.alpha_level
    d_ci_width = pooled["cohens_d_ci_hi"] - pooled["cohens_d_ci_lo"]

    # Does the effect clear the band the CONTROLS define? A binding score is
    # positive-means-binding, so a real effect sits ABOVE the band; anything
    # inside it is what a strength-matched control produces. Observed on
    # Gemma-3-27B-IT at three coefficients (64/128/256): pooled means -0.013,
    # -0.057, -0.181 with |d| up to 0.68 and every construction
    # Holm-significant, yet every one inside its own null band and within 20%
    # of the NULL_NON_PARTICIPANT control -- at c=256 the absent-entity push
    # moved the score MORE than the real one (ratio 0.84).
    pooled_mean = float(pooled["mean"])
    clears_null_band = not (band_lo <= pooled_mean <= band_hi)
    strongest_control = (
        min(table.null_by_edit, key=lambda e: float(np.mean(table.null_by_edit[e])))
        if table.null_by_edit
        else None
    )
    control_ratio = None
    if strongest_control is not None:
        strongest_mean = float(np.mean(table.null_by_edit[strongest_control]))
        if abs(strongest_mean) > 1e-12:
            control_ratio = pooled_mean / strongest_mean

    if not strength_passes:
        outcome = "uninterpretable_strength_failure"
    elif pooled_significant and not clears_null_band:
        # Ordered BEFORE the effect-size branches on purpose: p and d both
        # answer "how reliably", and neither answers "vs what".
        outcome = "indistinguishable_from_controls"
    elif (
        pooled_significant
        and pooled["cohens_d"] >= _D_MEANINGFUL
        and pooled_mean > band_hi
        and len(meaningful) >= 2
        and random_in_band
    ):
        outcome = "positive_binding"
    elif pooled_significant and pooled["cohens_d"] < _D_MEANINGFUL:
        outcome = "significant_but_tiny"
    elif pooled_significant:
        outcome = "suggestive_not_conclusive"
    elif d_ci_width <= _D_CI_TIGHT:
        outcome = "clean_negative"
    else:
        outcome = "inconclusive_underpowered"

    return {
        "outcome": outcome,
        "strength_check_passes": strength_passes,
        "strength_gated_on": "role_push",
        # Reported, NOT gating: the addressability control. Its failure means
        # the IDENTITY_SWAP claim is unsupported, not that the push result is.
        "identity_swap_check_passes": bool(neutral_check.get("available"))
        and bool(neutral_check.get("passes")),
        "pooled_significant": bool(pooled_significant),
        # The comparison the outcome now turns on, reported so a reader never
        # has to recompute it from control_means to know what the label means.
        "pooled_mean": pooled_mean,
        "null_band": [band_lo, band_hi],
        "clears_null_band": bool(clears_null_band),
        "strongest_control": strongest_control,
        "vs_strongest_control_ratio": control_ratio,
        "pooled_d": pooled["cohens_d"],
        "pooled_d_ci": [pooled["cohens_d_ci_lo"], pooled["cohens_d_ci_hi"]],
        "d_threshold": _D_MEANINGFUL,
        "meaningful_constructions": meaningful,
        "random_direction_mean": random_mean,
        "random_direction_in_band": random_in_band,
        "control_means": {
            edit: float(np.mean(scores)) for edit, scores in table.null_by_edit.items()
        },
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
