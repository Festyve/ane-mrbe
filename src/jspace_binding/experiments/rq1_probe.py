"""RQ1: is the target entity's role linearly decodable, and from where?

Caches no-edit activations for every primary-stimulus cell via
model.probe_activation and runs the leave-one-pair-out ridge probe per
(injection site x activation source), reporting accuracy, control-task
accuracy, and selectivity (analysis.probes). The three sources — J-space
component, orthogonal remainder, full residual — localize any decodable role
signal relative to the workspace.

This is the proposal's warm-up analysis: decodability is evidence about
information PRESENCE, not use (RQ2's ablation speaks to use), and a linear
null is nearly uninformative (Smolensky-style multiplicative codes are
invisible to it). analyze-time interpretation stays in those bounds.
"""

from __future__ import annotations

import json
from pathlib import Path

from jspace_binding.analysis.plots import selectivity_plot
from jspace_binding.analysis.probes import (
    PROBE_SOURCES,
    ProbeExample,
    ProbeReport,
    leave_one_pair_out,
)
from jspace_binding.config import Config
from jspace_binding.model.base import ProbeActivationSource
from jspace_binding.types import InjectionSite, ItemFamily, Position, Role


def run_rq1(
    config: Config, model: ProbeActivationSource, families: list[ItemFamily]
) -> dict[str, object]:
    """Collect activations, probe every (site, source), plot, summarize.

    Labels come from the design (the cell's Role), never inferred from the
    sentence. The figure and JSON land beside the primary experiment's
    outputs (config.paths.figures / results).
    """
    reports: dict[InjectionSite, dict[str, ProbeReport]] = {}
    for site in config.experiment.injection_sites:
        examples: dict[str, list[ProbeExample]] = {source: [] for source in PROBE_SOURCES}
        for family in families:
            entity = family.concept_pair.entity
            for role in Role:
                for position in Position:
                    sentence = family.cell(role, position).sentence
                    activations = model.probe_activation(sentence, entity, site)
                    missing = [s for s in PROBE_SOURCES if s not in activations]
                    if missing:
                        raise ValueError(
                            f"probe_activation missing sources {missing} for {sentence!r}"
                        )
                    for source in PROBE_SOURCES:
                        examples[source].append(
                            ProbeExample(
                                pair_id=family.concept_pair.pair_id,
                                is_agent=role is Role.AGENT,
                                features=tuple(float(x) for x in activations[source]),
                            )
                        )
        reports[site] = {
            source: leave_one_pair_out(rows, control_seed=config.experiment.seed)
            for source, rows in examples.items()
        }

    figures_dir = Path(config.paths.figures)
    figure_path = figures_dir / "rq1_selectivity.png"
    selectivity_plot(reports, figure_path)

    summary: dict[str, object] = {
        "sites": {
            site.value: {
                source: {
                    "accuracy": report.accuracy,
                    "control_accuracy": report.control_accuracy,
                    "selectivity": report.selectivity,
                    "n_examples": report.n_examples,
                    "n_folds": report.n_folds,
                }
                for source, report in by_source.items()
            }
            for site, by_source in reports.items()
        },
        "figure": str(figure_path),
        "note": (
            "decodability, not use; a linear null does not rule out "
            "multiplicative binding codes (proposal, Potential Limitations)"
        ),
    }
    out = Path(config.paths.results) / "rq1_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
