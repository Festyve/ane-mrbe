"""RQ2: does the model USE the workspace for binding? Ablation deltas.

For every primary-stimulus cell, run three conditions — NO_EDIT,
ABLATE_JSPACE (remove the J-space component at the site), and
ABLATE_RANDOM_SUBSPACE (remove a random subspace of matched dimension, the
capacity control) — under both probes, and score two tasks:

- binding task (ROLE probe): correct iff the participant the model ranks
  higher is the sentence's actual agent — P(entity) > P(other) exactly when
  the entity is the agent.
- recall task (NEUTRAL probe): correct iff both mentioned participants
  outrank the absent counterpart token — min(P(entity), P(other)) >
  P(counterpart).

Both tasks are scored on the SAME sentences (identical token counts and
entity counts), so a selective binding deficit cannot be blamed on longer or
busier inputs. Baseline accuracy matching is NOT guaranteed by that and is
measured per run: `difficulty_matched` reports whether the two no-edit
baselines actually landed within the proposal's +/-2%. When they do not — in
particular when the recall baseline sits at ceiling — the recall term cannot
move, the binding-minus-recall subtraction degenerates to the raw binding
deficit, and `recall_control_informative` says so.

The binding-specific deficit for an ablation is

    (binding degradation) - (recall degradation),

and the causal-involvement claim requires the J-space bar to exceed both
zero and the random-subspace bar; if the two bars match, the degradation is
generic capacity loss (proposal, §4 Ablation).

Deficits carry a family-level percentile bootstrap CI (analysis.stats): the
point estimate alone cannot distinguish a real null from an intervention that
never landed. `edit_magnitude` reports the relative residual-norm change the
hooks actually applied, so those two cases stay separable — a deficit of zero
next to an edit magnitude of zero is a plumbing result, not a finding.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jspace_binding.analysis.plots import ablation_deltas_plot
from jspace_binding.analysis.stats import bootstrap_ci
from jspace_binding.config import Config
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
_SCORED_PROBES: tuple[ProbeKind, ...] = (ProbeKind.ROLE, ProbeKind.NEUTRAL)

# Baselines within this gap count as difficulty-matched (proposal, §5).
_DIFFICULTY_TOLERANCE = 0.02
# A no-edit baseline above this cannot fall far enough to serve as a control.
_CEILING = 0.99


def _drain_edit_magnitudes(model: WorkspaceModel) -> list[float]:
    """Relative residual-norm changes recorded since the last drain.

    Optional backend capability: DummyModel does not apply residual-stream
    edits at all, so absence is reported as "unavailable" rather than zero —
    zero would be indistinguishable from a real no-op edit.
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
        # An edit that moves the residual this little cannot plausibly change
        # a downstream argmax; a null deficit beside it is uninformative.
        "edit_landed": bool(arr.mean() > 1e-4),
    }


def _run_site(
    config: Config,
    model: WorkspaceModel,
    families: list[ItemFamily],
    site: InjectionSite,
) -> dict[str, Any]:
    """Score both tasks under all three conditions at one injection site."""
    # Per-family accuracy, so resampling can happen at the family level (the
    # unit analysis.stats is written around) rather than over correlated cells.
    per_family: dict[tuple[EditType, ProbeKind], list[float]] = {
        (edit_type, probe): [] for edit_type in _CONDITIONS for probe in _SCORED_PROBES
    }
    magnitudes: dict[EditType, list[float]] = {edit: [] for edit in ABLATIONS}

    for family_index, family in enumerate(families):
        answers = family.answer_set
        if answers is None:
            raise ValueError(f"family {family.family_id!r} has no answer_set")
        for edit_type in _CONDITIONS:
            edit = EditSpec(
                edit_type=edit_type,
                # Fresh, reproducible random subspace per family (NO_EDIT and
                # ABLATE_JSPACE carry no randomness; the seed is inert there).
                seed=config.experiment.seed + family_index,
            )
            cells: dict[ProbeKind, list[bool]] = {probe: [] for probe in _SCORED_PROBES}
            for role in Role:
                for position in Position:
                    sentence = family.cell(role, position).sentence
                    for probe_kind, probe in (
                        (ProbeKind.ROLE, family.role_probe),
                        (ProbeKind.NEUTRAL, family.neutral_probe),
                    ):
                        probs = model.answer_distribution(
                            sentence, probe, edit, site, answers.tokens
                        )
                        if probe_kind is ProbeKind.ROLE:
                            picked_entity = probs[answers.entity] > probs[answers.other]
                            hit = picked_entity == (role is Role.AGENT)
                        else:
                            hit = (
                                min(probs[answers.entity], probs[answers.other])
                                > probs[answers.counterpart]
                            )
                        cells[probe_kind].append(bool(hit))
            for probe_kind, hits in cells.items():
                per_family[(edit_type, probe_kind)].append(float(np.mean(hits)))
            if edit_type in magnitudes:
                magnitudes[edit_type].extend(_drain_edit_magnitudes(model))

    accuracy = {key: float(np.mean(vals)) for key, vals in per_family.items() if vals}
    base_role = np.asarray(per_family[(EditType.NO_EDIT, ProbeKind.ROLE)])
    base_neutral = np.asarray(per_family[(EditType.NO_EDIT, ProbeKind.NEUTRAL)])
    baseline_gap = abs(float(base_role.mean() - base_neutral.mean()))

    deltas: dict[str, Any] = {}
    for ablation in ABLATIONS:
        # Per-family deficits keep the no-edit/ablation pairing inside each
        # resampled unit, so the bootstrap CI is over the paired difference.
        binding = base_role - np.asarray(per_family[(ablation, ProbeKind.ROLE)])
        recall = base_neutral - np.asarray(per_family[(ablation, ProbeKind.NEUTRAL)])
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
        # A recall baseline at ceiling cannot register damage, so subtracting
        # it is a no-op and "binding-specific" means nothing stronger than
        # "binding". Reported so a degenerate control is never read as a passed one.
        "recall_control_informative": bool(base_neutral.mean() < _CEILING),
        "deltas": deltas,
        # Proposal, §4: causal involvement requires binding to degrade more
        # than recall AND more than the matched random subspace predicts. The
        # CI must also clear zero — a point estimate of a few trials in 2400 is
        # not a deficit.
        "workspace_causally_involved": bool(
            jspace["binding_specific_deficit"] > 0.0
            and jspace["binding_specific_deficit"] > random_sub["binding_specific_deficit"]
            and jspace["ci_excludes_zero"]
        ),
        # False = the null is uninterpretable, not evidence against involvement.
        "null_interpretable": bool(
            (magnitude is None or magnitude["edit_landed"])
            and base_neutral.mean() < _CEILING
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

    Defaults to config.experiment.injection_sites rather than a single
    hard-coded site: ablating where no role signal was found (RQ1) and
    reporting no effect is close to tautological, and which site carries the
    signal is an empirical question RQ1 answers per run. `site` is retained as
    a single-site override.
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
        "provenance": run_provenance(config, n_families=len(families)),
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
