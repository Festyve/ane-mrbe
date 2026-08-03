"""RQ2: does the model USE the workspace for binding? Ablation deltas.

For every primary-stimulus cell, run three conditions — NO_EDIT,
ABLATE_JSPACE (remove the J-space component at the site), and
ABLATE_RANDOM_SUBSPACE (remove a random subspace of matched dimension, the
capacity control) — under both probes, and score two tasks:

- binding task (ROLE probe): the signed log-odds gap between the sentence's
  actual agent and the other participant, `logit(P(entity)) -
  logit(P(other))` (sign-flipped when the entity is the patient) — positive
  and large means confidently correct, matching P(entity) > P(other) at the
  zero crossing but continuous on both sides of it.
- recall task (CONCEPT probe): the log-odds gap favouring whichever
  participant matches the probe's profession cue, `logit(P(correct)) -
  logit(P(incorrect))`, COUNTERBALANCED — asked once for each participant's
  cue and averaged.

Why CONCEPT and not NEUTRAL. NEUTRAL asks "which professions are mentioned",
pitting two in-context words against one that never appears. No ablation
small enough to be informative about binding can close that gap, so the
control sat at exactly 1.0 under every condition — in DummyModel as well as
on the real model — and the binding-minus-recall subtraction reduced to the
raw binding deficit. CONCEPT puts BOTH candidate answers in the sentence, so
lexical presence cannot answer it and the model has to know what the
profession is. Measured on Qwen2.5-1.5B over the committed stimuli (n=80
active_passive families, scored as below): NEUTRAL +3.77 log-odds versus
CONCEPT +1.42, positive on 80/80 families. Same units, direct comparison —
CONCEPT keeps the model clearly correct while leaving room to fall. Reproduce
with `python scripts/check_concept_probe.py --limit 80`; the full write-up,
including per-pair cue strength, is docs/CONCEPT_PROBE.md.

Counterbalancing is load-bearing, not cosmetic. One-sided semantic probes
are confounded by base rate and by primacy — reordering identical tokens
moved a one-sided probe by 3.6 log-odds. Averaging the two askings cancels
both, because each confound favours the entity in one asking and the other
participant in the other.

CONCEPT is role-blind by construction — its answer is invariant under the
agent/patient swap — and measured to be so: the signed agent-minus-patient
shift is +0.054 against a SEM of 0.035, i.e. 1.5 SEM and 3.8% of the margin,
so what movement exists is per-cell noise that averaging the four
role x position cells removes. NEUTRAL is still built and still used by
primary.py for the IDENTITY_SWAP strength check, which needs its
present-vs-absent contrast; only RQ2's recall control moves to CONCEPT.

Both tasks are scored in log-odds rather than raw probability for the reason
analysis.binding_score already uses it: a threshold on raw probability reads
"correct" regardless of how much confidence eroded underneath, so partial
damage is invisible until the argmax flips. A plain pass/fail accuracy is
still reported per probe (`accuracy`) for eyeballing and for
`difficulty_matched`, but the deficits below are computed from the margin.

Both tasks are scored on the SAME sentences (identical token counts and
entity counts), so a selective binding deficit cannot be blamed on longer or
busier inputs. Baseline accuracy matching is NOT guaranteed by that and is
measured per run: `difficulty_matched` reports whether the two no-edit
baselines actually landed within the proposal's +/-2%.

The binding-specific deficit for an ablation is

    (binding margin degradation) - (recall margin degradation),

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
    """Signed log-odds gap: positive means the model favors the sentence's
    actual agent, magnitude is confidence. Continuous analogue of
    `picked_entity == (role is AGENT)` — agrees with it at the zero
    crossing, but registers a partial shift even when the argmax does not
    flip."""
    gap = logit(probs[entity]) - logit(probs[other])
    return gap if role is Role.AGENT else -gap


def _concept_margin(probs: dict[str, float], correct: str, incorrect: str) -> float:
    """Log-odds gap favouring the participant that matches the probe's cue.

    Both are present in the sentence, so this cannot be answered from lexical
    presence — which is precisely what the NEUTRAL probe could be, and why it
    sat at ceiling. One asking is confounded by base rate and primacy; the
    caller averages the two askings, which cancels both.
    """
    return logit(probs[correct]) - logit(probs[incorrect])


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


# Ablation must damage binding by at least this SHARE OF ITS OWN BASELINE
# MARGIN before a causal claim is on the table. Deficits are already
# normalised per task, so this reads directly as "5% of the no-edit margin".
#
# Sign alone is not enough. On Gemma-3-12B at final_token the observed
# binding_deficit was +0.00058 against a baseline margin of 2.56 -- 0.02% of
# it, indistinguishable from noise, yet strictly positive, so a `> 0` test
# passed and the run reported causal involvement. The number that cleared the
# CI there was the CONTROL task drifting (-0.0074), not binding being damaged.
_MIN_BINDING_DEFICIT = 0.05


def _involvement_verdict(
    binding_deficit: float,
    binding_specific: float,
    random_specific: float,
    ci_excludes_zero: bool,
) -> bool:
    """Does this ablation support causal involvement of the workspace?

    The binding_deficit clause is load-bearing, and needs a MAGNITUDE, not just
    a sign. `binding - recall` is a SELECTIVITY measure: it presupposes that
    ablation damaged binding, and it stays large whenever the two tasks merely
    move apart. So the verdict has to establish damage before reading the
    subtraction at all.

    Two observed failures, both of which this guards:

    - Qwen3.6-27B, entity_token: binding -0.032, recall -0.046, difference
      +0.014 with a CI clearing zero. Ablation IMPROVED both tasks; the
      positive difference only said recall improved more. Caught by the sign.
    - Gemma-3-12B, final_token: binding +0.00058 (0.02% of a 2.56 baseline
      margin), recall -0.0074, difference +0.0079 with a CI clearing zero and
      beating the random control. Binding was untouched and the CONTROL drifted;
      the difference was made entirely of the control. A sign test passes this.
      Caught by _MIN_BINDING_DEFICIT.

    Both share the failure this project has now hit four times: comparing two
    quantities without first asking whether either is distinguishable from
    nothing.
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
    # Per-family accuracy (pass/fail, for reporting) and margin (log-odds,
    # for the deficits below), so resampling can happen at the family level
    # (the unit analysis.stats is written around) rather than over
    # correlated cells.
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
                # Fresh, reproducible random subspace per family (NO_EDIT and
                # ABLATE_JSPACE carry no randomness; the seed is inert there).
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

                    # CONCEPT, counterbalanced: ask once with each participant's
                    # cue and average. Each asking carries the base-rate and
                    # primacy advantage in the opposite direction, so the mean
                    # is free of both; either asking alone is not.
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

    # Deficits are expressed as a FRACTION of each task's own no-edit baseline
    # margin, not in raw log-odds. The two tasks sit at very different
    # magnitudes — the role gap is several log-odds, the concept gap is well
    # under one by design — so a raw subtraction is dimensionally wrong:
    # under `bag` ground truth BOTH collapse completely, yet
    # (4.0 - 0.7) still reads as a large binding-specific deficit and the run
    # wrongly reports causal involvement. Dividing each by its own baseline
    # puts both on a "share of baseline discriminability lost" scale where 1.0
    # means total collapse, so two total collapses cancel to ~0 as the design
    # requires. (The old pass/fail version avoided this only by accident:
    # accuracy is bounded, so both tasks happened to share a 0-1 scale.)
    #
    # Normalised by the MEAN baseline rather than per family: a single family
    # whose baseline is near zero would otherwise blow up its own ratio, while
    # the mean keeps per-family variation intact for the bootstrap.
    role_scale = float(base_role_margin.mean())
    recall_scale = float(base_recall_margin.mean())

    # A task the model cannot do at baseline has no discriminability to lose,
    # so a deficit measured against it means nothing. This is the successor to
    # the ceiling check: NEUTRAL failed by being too easy, and a probe can fail
    # equally by being too hard.
    recall_informative = bool(
        recall_scale > _MIN_BASELINE_MARGIN
        and float(np.std(base_recall_margin)) > _MARGIN_SPREAD_EPS
    )
    binding_baseline_usable = bool(role_scale > _MIN_BASELINE_MARGIN)

    deltas: dict[str, Any] = {}
    for ablation in ABLATIONS:
        # Per-family deficits keep the no-edit/ablation pairing inside each
        # resampled unit, so the bootstrap CI is over the paired difference.
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
        # Reporting only: true whenever recall pass/fail is trivially easy.
        # No longer gates interpretability — see recall_control_informative.
        "recall_accuracy_at_ceiling": bool(base_recall_acc.mean() >= _CEILING),
        "recall_control_informative": recall_informative,
        "deltas": deltas,
        # Proposal, §4: causal involvement requires binding to degrade more
        # than recall AND more than the matched random subspace predicts. The
        # CI must also clear zero — a point estimate of a few trials in 2400 is
        # not a deficit.
        #
        # `binding_deficit > 0` is the load-bearing addition. The subtraction
        # `binding − recall` is only a SELECTIVITY measure; it presupposes that
        # ablation damaged something. When BOTH deficits come out negative —
        # i.e. ablation IMPROVED both tasks — a positive difference just means
        # recall improved more than binding did, which is not evidence that the
        # workspace is causally involved in binding.
        #
        # Observed on Qwen3.6-27B at entity_token: binding −0.032, recall
        # −0.046, difference +0.014, CI clearing zero. The old rule reported
        # causal involvement from two improvements. Same failure shape as the
        # E4 `always_on` label: a threshold that never asked which side of zero
        # the inputs were on.
        "workspace_causally_involved": _involvement_verdict(
            binding_deficit=jspace["binding_deficit"],
            binding_specific=jspace["binding_specific_deficit"],
            random_specific=random_sub["binding_specific_deficit"],
            ci_excludes_zero=jspace["ci_excludes_zero"],
        ),
        # Non-null when ablation IMPROVED a task instead of damaging it. Both
        # deficits negative makes the selectivity subtraction uninterpretable,
        # so it is surfaced rather than silently folded into the verdict.
        "ablation_improved_performance": {
            "binding": bool(jspace["binding_deficit"] < 0.0),
            "recall": bool(jspace["recall_deficit"] < 0.0),
        },
        # The random control is matched on RANK (ablate_k directions), not on
        # perturbation magnitude, and on this model the two differ by ~5x
        # (0.234 vs 0.043 mean relative norm change) because J-space ablation
        # removes the most strongly active directions while the control removes
        # arbitrary ones. A jspace effect exceeding the control is therefore not
        # by itself evidence of localisation — it may only reflect the larger
        # perturbation. Reported so the comparison is never read as
        # magnitude-matched.
        "edit_magnitude_ratio": (
            None
            if not (jspace["edit_magnitude"] and random_sub["edit_magnitude"])
            else jspace["edit_magnitude"]["mean_relative_norm_change"]
            / max(random_sub["edit_magnitude"]["mean_relative_norm_change"], 1e-12)
        ),
        # Baseline discriminability both deficits are expressed as a share of.
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
