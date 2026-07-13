"""Primary experiment: the full condition sweep and the analysis behind the paper numbers.

run_primary is written strictly against the WorkspaceModel protocol, so the
identical sweep drives the GPU-free DummyModel (validating the analysis against
known ground truth) and, later, the real Qwen + J-lens backend. analyze() turns
the trial list into per-(construction, pair) binding scores with bootstrap CIs,
permutation p-values (Holm-corrected across groups), Cohen's d, the control
null band, and both figures.
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
    EditType,
    InjectionSite,
    ItemFamily,
    Position,
    ProbeKind,
    Role,
    TrialResult,
)


def run_primary(
    config: Config, model: WorkspaceModel, families: list[ItemFamily]
) -> list[TrialResult]:
    """Sweep family x (role x position) cell x edit type x injection site.

    Every condition runs the ROLE probe; REAL edits additionally run the
    NEUTRAL probe — the intervention-strength control showing the edit
    propagated without asking about roles. Trials are written to
    config.paths.results / "trials.jsonl" and returned in sweep order.
    """
    trials: list[TrialResult] = []
    for family_index, family in enumerate(families):
        if family.answer_set is None:
            raise ValueError(f"family {family.family_id!r} has no answer_set")
        for edit_type in config.experiment.edit_types:
            edit = plan_edit(
                family,
                edit_type,
                non_participant=config.stimuli.non_participant_concept,
                alpha=config.model.alpha,
                # Offset per family so RANDOM_DIRECTION draws a fresh, reproducible
                # vector per family instead of reusing one direction everywhere.
                seed=config.experiment.seed + family_index,
            )
            probe_kinds = (
                (ProbeKind.ROLE, ProbeKind.NEUTRAL)
                if edit_type is EditType.REAL
                else (ProbeKind.ROLE,)
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
                                stimulus.sentence, probe, edit, site, family.answer_set.tokens
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
                                        token: float(p) for token, p in answer_probs.items()
                                    },
                                )
                            )
    _write_trials(trials, Path(config.paths.results) / "trials.jsonl")
    return trials


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
        raise ValueError("analyze: no REAL-edit ROLE-probe trials at the final-token site")
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
    # The per-condition figure reads one pair's target token; the sweep's first
    # trial makes the choice deterministic and guaranteed present in the data.
    example_target = trials[0].pair_id.split("->", 1)[1]
    per_condition_plot(trials, example_target, per_condition_path)

    return {
        "site": site.value,
        "ci_level": float(config.analysis.ci_level),
        "n_trials": len(trials),
        "null_band": [float(band_lo), float(band_hi)],
        "n_null_scores": len(table.null_band),
        "pooled": pooled,
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
            }
            f.write(json.dumps(record) + "\n")
