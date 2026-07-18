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

Task difficulty matching (proposal, §5 task-difficulty confound): both tasks
are scored on the SAME sentences — identical token counts, entity counts,
and (by construction) matched no-edit baseline accuracy — so a selective
binding deficit cannot be blamed on the binding task being harder. The
binding-specific deficit for an ablation is

    (binding degradation) - (recall degradation),

and the causal-involvement claim requires the J-space bar to exceed both
zero and the random-subspace bar; if the two bars match, the degradation is
generic capacity loss (proposal, §4 Ablation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jspace_binding.analysis.plots import ablation_deltas_plot
from jspace_binding.config import Config
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


def run_rq2(
    config: Config,
    model: WorkspaceModel,
    families: list[ItemFamily],
    site: InjectionSite = InjectionSite.FINAL_TOKEN,
) -> dict[str, object]:
    """Score both tasks under all three conditions; plot and summarize."""
    correct: dict[tuple[EditType, ProbeKind], list[bool]] = {
        (edit_type, probe): [] for edit_type in _CONDITIONS for probe in ProbeKind
    }
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
                        correct[(edit_type, probe_kind)].append(bool(hit))

    accuracy = {
        key: float(np.mean(values)) for key, values in correct.items() if values
    }
    baseline_gap = abs(
        accuracy[(EditType.NO_EDIT, ProbeKind.ROLE)]
        - accuracy[(EditType.NO_EDIT, ProbeKind.NEUTRAL)]
    )
    deltas = {}
    for ablation in ABLATIONS:
        binding_deficit = (
            accuracy[(EditType.NO_EDIT, ProbeKind.ROLE)]
            - accuracy[(ablation, ProbeKind.ROLE)]
        )
        recall_deficit = (
            accuracy[(EditType.NO_EDIT, ProbeKind.NEUTRAL)]
            - accuracy[(ablation, ProbeKind.NEUTRAL)]
        )
        deltas[ablation.value] = {
            "binding_deficit": binding_deficit,
            "recall_deficit": recall_deficit,
            "binding_specific_deficit": binding_deficit - recall_deficit,
        }

    figures_dir = Path(config.paths.figures)
    figure_path = figures_dir / "rq2_ablation_deltas.png"
    ablation_deltas_plot(deltas, figure_path)

    jspace = deltas[EditType.ABLATE_JSPACE.value]["binding_specific_deficit"]
    random_sub = deltas[EditType.ABLATE_RANDOM_SUBSPACE.value]["binding_specific_deficit"]
    summary: dict[str, object] = {
        "site": site.value,
        "accuracy": {
            f"{edit_type.value}|{probe.value}": acc
            for (edit_type, probe), acc in accuracy.items()
        },
        "baseline_task_accuracy_gap": baseline_gap,
        "difficulty_matched": bool(baseline_gap <= 0.02),  # proposal: ±2% baselines
        "deltas": deltas,
        # Proposal, §4: causal involvement requires binding to degrade more
        # than recall AND more than the matched random subspace predicts.
        "workspace_causally_involved": bool(jspace > 0.0 and jspace > random_sub),
        "figure": str(figure_path),
    }
    out = Path(config.paths.results) / "rq2_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
