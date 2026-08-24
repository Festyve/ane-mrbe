"""RQ2: does the model USE the workspace for binding? Ablation deltas.

For every primary-stimulus cell, run three conditions — NO_EDIT, ABLATE_JSPACE,
and ABLATE_RANDOM_SUBSPACE (the rank-matched capacity control) — and score two
tasks in log-odds: binding (ROLE probe, signed agent-vs-patient gap) and recall
(CONCEPT probe, counterbalanced over both participants' cues). The
binding-specific deficit is their difference, and causal involvement requires
the J-space bar to clear zero, the random-subspace bar, and a magnitude floor.

CONCEPT replaced the NEUTRAL recall control because NEUTRAL sat at exactly 1.0
under every condition and so could not register damage; docs/CONCEPT_PROBE.md
has the validation. Counterbalancing cancels base-rate and primacy confounds,
which move a one-sided probe by up to 3.6 log-odds.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jspace_binding.analysis.binding_score import logit
from jspace_binding.analysis.plots import ablation_deltas_plot
from jspace_binding.analysis.stats import bootstrap_ci
from jspace_binding.config import Config
from jspace_binding.experiments.progress import track
from jspace_binding.experiments.provenance import run_provenance
from jspace_binding.model.base import WorkspaceModel
from jspace_binding.types import (
    EditSpec,
    EditType,
    InjectionSite,
    ItemFamily,
    Position,
    ProbeKind,
    Role,
)

ABLATIONS: tuple[EditType, ...] = (EditType.ABLATE_JSPACE, EditType.ABLATE_RANDOM_SUBSPACE)
_CONDITIONS: tuple[EditType, ...] = (EditType.NO_EDIT, *ABLATIONS)
# RQ2 scores the binding task against the recall task only. ProbeKind.RECIPIENT
# is dative-specific and not every family carries it, so it is named out rather
# than iterated over and silently left empty.
_SCORED_PROBES: tuple[ProbeKind, ...] = (ProbeKind.ROLE, ProbeKind.CONCEPT)

# Baselines within this gap count as difficulty-matched (proposal, §5).
_DIFFICULTY_TOLERANCE = 0.02
# Reporting-only: a no-edit accuracy this high means the pass/fail readout
# cannot register damage. No longer gates interpretability (the margin can
# still move below this), kept so a trivially-easy recall task is visible.
_CEILING = 0.99
# Family-level margin baseline must show real spread, not a frozen constant
# (e.g. every trial saturating the logit clamp), to count as informative.
_MARGIN_SPREAD_EPS = 1e-6
# Baseline log-odds margin below which a task is not reliably doable, so the
# share-of-baseline deficits would divide by noise. 0.2 log-odds ~ 55/45; the
# CONCEPT probe measures +1.42 overall on Qwen2.5-1.5B and its weakest
# profession pair still reaches +0.61, so every pair clears this comfortably.
_MIN_BASELINE_MARGIN = 0.2


def _role_margin(probs: dict[str, float], entity: str, other: str, role: Role) -> float:
    """Signed log-odds gap; positive favours the sentence's actual agent.

    Continuous analogue of `picked_entity == (role is AGENT)`: it registers a
    partial shift even when the argmax does not flip.
    """
    gap = logit(probs[entity]) - logit(probs[other])
    return gap if role is Role.AGENT else -gap


def _concept_margin(probs: dict[str, float], correct: str, incorrect: str) -> float:
    """Log-odds gap favouring the participant matching the probe's cue.

    Both are present in the sentence, so lexical presence cannot answer it. The
    caller averages the two askings to cancel base-rate and primacy confounds.
    """
    return logit(probs[correct]) - logit(probs[incorrect])


def _drain_edit_magnitudes(model: WorkspaceModel) -> list[float]:
    """Relative residual-norm changes since the last drain.

    Optional backend capability. Absence is reported as unavailable rather than
    zero, which would be indistinguishable from a real no-op edit.
    """
    drain = getattr(model, "drain_edit_magnitudes", None)
    return list(drain()) if callable(drain) else []


def _magnitude_summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "mean_relative_norm_change": float(arr.mean()),
        "median_relative_norm_change": float(np.median(arr)),
        "max_relative_norm_change": float(arr.max()),
        "n_edits_applied": int(arr.size),
        # Below this the edit cannot plausibly move a downstream argmax.
        "edit_landed": bool(arr.mean() > 1e-4),
    }


# Minimum binding damage, as a share of the task's own no-edit margin, before a
# causal claim is on the table. A sign test is not enough: Gemma-3-12B at
# final_token gave +0.00058 against a 2.56 baseline and still passed `> 0`.
_MIN_BINDING_DEFICIT = 0.05


def _involvement_verdict(
    binding_deficit: float,
    binding_specific: float,
    random_specific: float,
    ci_excludes_zero: bool,
) -> bool:
    """Does this ablation support causal involvement of the workspace?

    `binding - recall` is a selectivity measure that presupposes damage, so the
    binding_deficit clause has to establish damage before the subtraction is
    read at all. RESULTS.md "Note on verdict labels" records the two runs that
    reported involvement without it.
    """
    return bool(
        binding_deficit > _MIN_BINDING_DEFICIT
        and binding_specific > 0.0
        and binding_specific > random_specific
        and ci_excludes_zero
    )


def _run_site(
    config: Config,
    model: WorkspaceModel,
    families: list[ItemFamily],
    site: InjectionSite,
) -> dict[str, Any]:
    """Score both tasks under all three conditions at one injection site."""
    # Kept per family so the bootstrap resamples families, not correlated cells.
    per_family_accuracy: dict[tuple[EditType, ProbeKind], list[float]] = {
        (edit_type, probe): [] for edit_type in _CONDITIONS for probe in _SCORED_PROBES
    }
    per_family_margin: dict[tuple[EditType, ProbeKind], list[float]] = {
        (edit_type, probe): [] for edit_type in _CONDITIONS for probe in _SCORED_PROBES
    }
    magnitudes: dict[EditType, list[float]] = {edit: [] for edit in ABLATIONS}

    for family_index, family in enumerate(
        track(families, f"rq2 {site.value}", total=len(families))
    ):
        answers = family.answer_set
        if answers is None:
            raise ValueError(f"family {family.family_id!r} has no answer_set")
        if not (family.concept_probe_entity and family.concept_probe_other):
            raise ValueError(
                f"family {family.family_id!r} has no CONCEPT probes; RQ2's recall "
                "control needs both askings for counterbalancing. Regenerate the "
                "stimuli (scripts/generate_stimuli.py) — a file written before "
                "ProbeKind.CONCEPT existed will load with them empty."
            )
        for edit_type in _CONDITIONS:
            edit = EditSpec(
                edit_type=edit_type,
                # Fresh, reproducible random subspace per family; inert elsewhere.
                seed=config.experiment.seed + family_index,
            )
            cells: dict[ProbeKind, list[bool]] = {probe: [] for probe in _SCORED_PROBES}
            margin_cells: dict[ProbeKind, list[float]] = {probe: [] for probe in _SCORED_PROBES}
            for role in Role:
                for position in Position:
                    sentence = family.cell(role, position).sentence
                    probs = model.answer_distribution(
                        sentence, family.role_probe, edit, site, answers.tokens
                    )
                    picked_entity = probs[answers.entity] > probs[answers.other]
                    cells[ProbeKind.ROLE].append(bool(picked_entity == (role is Role.AGENT)))
                    margin_cells[ProbeKind.ROLE].append(
                        _role_margin(probs, answers.entity, answers.other, role)
                    )

                    # Counterbalanced: each asking carries the base-rate and
                    # primacy advantage in the opposite direction.
                    hits, margins = [], []
                    for probe, correct, incorrect in (
                        (family.concept_probe_entity, answers.entity, answers.other),
                        (family.concept_probe_other, answers.other, answers.entity),
                    ):
                        probs = model.answer_distribution(
                            sentence, probe, edit, site, answers.tokens
                        )
                        hits.append(probs[correct] > probs[incorrect])
                        margins.append(_concept_margin(probs, correct, incorrect))
                    cells[ProbeKind.CONCEPT].append(bool(all(hits)))
                    margin_cells[ProbeKind.CONCEPT].append(float(np.mean(margins)))
            for probe_kind, hits in cells.items():
                per_family_accuracy[(edit_type, probe_kind)].append(float(np.mean(hits)))
            for probe_kind, margins in margin_cells.items():
                per_family_margin[(edit_type, probe_kind)].append(float(np.mean(margins)))
            if edit_type in magnitudes:
                magnitudes[edit_type].extend(_drain_edit_magnitudes(model))

    accuracy = {
        key: float(np.mean(vals)) for key, vals in per_family_accuracy.items() if vals
    }
    base_role_acc = np.asarray(per_family_accuracy[(EditType.NO_EDIT, ProbeKind.ROLE)])
    base_recall_acc = np.asarray(per_family_accuracy[(EditType.NO_EDIT, ProbeKind.CONCEPT)])
    baseline_gap = abs(float(base_role_acc.mean() - base_recall_acc.mean()))

    base_role_margin = np.asarray(per_family_margin[(EditType.NO_EDIT, ProbeKind.ROLE)])
    base_recall_margin = np.asarray(per_family_margin[(EditType.NO_EDIT, ProbeKind.CONCEPT)])

    # Deficits are a share of each task's own no-edit margin, not raw log-odds:
    # the two tasks sit at very different magnitudes, so a raw subtraction reads
    # two total collapses as a large binding-specific deficit. Normalised by the
    # mean baseline, so a near-zero family cannot blow up its own ratio.
    role_scale = float(base_role_margin.mean())
    recall_scale = float(base_recall_margin.mean())

    # A task the model cannot do at baseline has no discriminability to lose.
    recall_informative = bool(
        recall_scale > _MIN_BASELINE_MARGIN
        and float(np.std(base_recall_margin)) > _MARGIN_SPREAD_EPS
    )
    binding_baseline_usable = bool(role_scale > _MIN_BASELINE_MARGIN)

    deltas: dict[str, Any] = {}
    for ablation in ABLATIONS:
        # Paired per family, so the CI is over the paired difference.
        binding = (
            base_role_margin - np.asarray(per_family_margin[(ablation, ProbeKind.ROLE)])
        ) / role_scale
        recall = (
            base_recall_margin - np.asarray(per_family_margin[(ablation, ProbeKind.CONCEPT)])
        ) / recall_scale
        specific = binding - recall
        lo, hi = bootstrap_ci(
            specific,
            n_resamples=config.analysis.n_bootstrap,
            ci_level=config.analysis.ci_level,
            seed=config.experiment.seed,
        )
        deltas[ablation.value] = {
            "binding_deficit": float(binding.mean()),
            "recall_deficit": float(recall.mean()),
            "binding_specific_deficit": float(specific.mean()),
            "binding_specific_ci": [lo, hi],
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "edit_magnitude": _magnitude_summary(magnitudes[ablation]),
        }

    jspace = deltas[EditType.ABLATE_JSPACE.value]
    random_sub = deltas[EditType.ABLATE_RANDOM_SUBSPACE.value]
    magnitude = jspace["edit_magnitude"]
    return {
        "site": site.value,
        "accuracy": {
            f"{edit_type.value}|{probe.value}": acc
            for (edit_type, probe), acc in accuracy.items()
        },
        "n_families": len(families),
        "baseline_task_accuracy_gap": baseline_gap,
        "difficulty_matched": bool(baseline_gap <= _DIFFICULTY_TOLERANCE),
        # Reporting only; recall_control_informative is what gates.
        "recall_accuracy_at_ceiling": bool(base_recall_acc.mean() >= _CEILING),
        "recall_control_informative": recall_informative,
        "deltas": deltas,
        # Proposal §4: binding must degrade more than recall AND more than the
        # matched random subspace, with a CI clearing zero.
        "workspace_causally_involved": _involvement_verdict(
            binding_deficit=jspace["binding_deficit"],
            binding_specific=jspace["binding_specific_deficit"],
            random_specific=random_sub["binding_specific_deficit"],
            ci_excludes_zero=jspace["ci_excludes_zero"],
        ),
        # Both negative makes the selectivity subtraction uninterpretable.
        "ablation_improved_performance": {
            "binding": bool(jspace["binding_deficit"] < 0.0),
            "recall": bool(jspace["recall_deficit"] < 0.0),
        },
        # The control is rank-matched, NOT magnitude-matched — the two differ
        # by ~5x here. Reported so the comparison is never read as the latter.
        "edit_magnitude_ratio": (
            None
            if not (jspace["edit_magnitude"] and random_sub["edit_magnitude"])
            else jspace["edit_magnitude"]["mean_relative_norm_change"]
            / max(random_sub["edit_magnitude"]["mean_relative_norm_change"], 1e-12)
        ),
        "baseline_margin": {
            "binding": role_scale,
            "recall": recall_scale,
            "recall_usable": recall_informative,
            "binding_usable": binding_baseline_usable,
        },
        # False = the null is uninterpretable, not evidence against involvement.
        "null_interpretable": bool(
            (magnitude is None or magnitude["edit_landed"])
            and recall_informative
            and binding_baseline_usable
        ),
    }


def run_rq2(
    config: Config,
    model: WorkspaceModel,
    families: list[ItemFamily],
    sites: Sequence[InjectionSite] | None = None,
    site: InjectionSite | None = None,
) -> dict[str, object]:
    """Score both tasks under all three conditions, at every configured site.

    Defaults to every configured site: ablating where RQ1 found no signal and
    reporting no effect is close to tautological. `site` overrides to just one.
    """
    if site is not None and sites is not None:
        raise ValueError("run_rq2: pass either `sites` or `site`, not both")
    chosen = (
        [site] if site is not None else list(sites or config.experiment.injection_sites)
    )
    if not chosen:
        raise ValueError("run_rq2: no injection sites configured")

    figures_dir = Path(config.paths.figures)
    by_site: dict[str, Any] = {}
    figures: dict[str, str] = {}
    for injection_site in chosen:
        result = _run_site(config, model, families, injection_site)
        figure_path = figures_dir / f"rq2_ablation_deltas_{injection_site.value}.png"
        ablation_deltas_plot(result["deltas"], figure_path)
        result["figure"] = str(figure_path)
        by_site[injection_site.value] = result
        figures[injection_site.value] = str(figure_path)

    summary: dict[str, object] = {
        "provenance": run_provenance(config, n_families=len(families), model=model),
        "sites": by_site,
        "figures": figures,
        "workspace_causally_involved": bool(
            any(r["workspace_causally_involved"] for r in by_site.values())
        ),
        "null_interpretable": bool(all(r["null_interpretable"] for r in by_site.values())),
    }
    out = Path(config.paths.results) / "rq2_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
