"""E4: is binding information RECRUITED on demand?

Present identical stimulus tokens under a role question ("Who treated
someone?") and a bag question ("Which professions are mentioned?") and measure
how decodable role is from J-space in each:

    recruitment_delta = accuracy(role question) - accuracy(bag question)

    always-on binding : both high, delta ~ 0
    recruited         : role high, bag ~ chance, delta LARGE
    absent from jspace: both ~ chance

RQ1 cannot make this distinction — it reads with no question in context, so
always-on and on-demand look identical to it.

The read position is forced, not chosen: the question follows the sentence, so
under a causal mask reading at FINAL_TOKEN or ENTITY_TOKEN would return
byte-identical activations for both conditions and the delta would be 0.0 by
construction. The read is at the final token of the full prompt, and for the
same reason there is no per-site breakdown. Probe machinery is RQ1's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jspace_binding.analysis.plots import recruitment_plot
from jspace_binding.analysis.probes import (
    PROBE_SOURCES,
    ProbeExample,
    ProbeReport,
    leave_one_pair_out,
)
from jspace_binding.config import Config
from jspace_binding.experiments.progress import track
from jspace_binding.experiments.provenance import run_provenance
from jspace_binding.model.base import RecruitmentActivationSource
from jspace_binding.types import ItemFamily, Position, Role

# The two question conditions. ROLE is role-diagnostic; BAG is the role-blind
# question the model can answer without binding anything.
ROLE_CONDITION = "role_question"
BAG_CONDITION = "bag_question"
CONDITIONS: tuple[str, ...] = (ROLE_CONDITION, BAG_CONDITION)

# Chance is 0.5 (binary agent/patient). A condition within this of chance is
# treated as carrying no decodable role information.
_CHANCE_TOLERANCE = 0.1
# Delta below this counts as "no recruitment", above as recruitment. A probe
# accuracy swing of 0.15 is well outside leave-one-pair-out noise at these
# sample sizes while staying short of the near-total swing the dummy plants.
_RECRUITMENT_THRESHOLD = 0.15


def run_e4(
    config: Config,
    model: RecruitmentActivationSource,
    families: list[ItemFamily],
) -> dict[str, object]:
    """Probe role decodability under both questions; report the delta.

    Labels come from the design (the cell's Role), never inferred from the
    sentence — same rule as RQ1.
    """
    examples: dict[tuple[str, str], list[ProbeExample]] = {
        (condition, source): [] for condition in CONDITIONS for source in PROBE_SOURCES
    }

    for family in track(families, "e4", total=len(families)):
        entity = family.concept_pair.entity
        if not family.role_probe or not family.neutral_probe:
            raise ValueError(
                f"family {family.family_id!r} lacks a role or neutral probe; E4 needs "
                "both questions to compare"
            )
        probes = {ROLE_CONDITION: family.role_probe, BAG_CONDITION: family.neutral_probe}
        for role in Role:
            for position in Position:
                sentence = family.cell(role, position).sentence
                for condition, probe in probes.items():
                    activations = model.recruitment_activation(sentence, probe, entity)
                    missing = [s for s in PROBE_SOURCES if s not in activations]
                    if missing:
                        raise ValueError(
                            f"recruitment_activation missing sources {missing} for "
                            f"{sentence!r} under {condition}"
                        )
                    for source in PROBE_SOURCES:
                        examples[(condition, source)].append(
                            ProbeExample(
                                pair_id=family.concept_pair.pair_id,
                                is_agent=role is Role.AGENT,
                                features=tuple(float(x) for x in activations[source]),
                            )
                        )

    reports: dict[tuple[str, str], ProbeReport] = {
        key: leave_one_pair_out(rows, control_seed=config.experiment.seed)
        for key, rows in examples.items()
    }

    by_source: dict[str, Any] = {}
    for source in PROBE_SOURCES:
        role_report = reports[(ROLE_CONDITION, source)]
        bag_report = reports[(BAG_CONDITION, source)]
        delta = role_report.accuracy - bag_report.accuracy
        by_source[source] = {
            ROLE_CONDITION: _report_block(role_report),
            BAG_CONDITION: _report_block(bag_report),
            "recruitment_delta": delta,
            "verdict": _verdict(role_report.accuracy, bag_report.accuracy, delta),
        }

    figures_dir = Path(config.paths.figures)
    figure_path = figures_dir / "e4_recruitment.png"
    recruitment_plot(by_source, figure_path)

    jspace = by_source["jspace"]
    summary: dict[str, object] = {
        "provenance": run_provenance(config, n_families=len(families), model=model),
        "sources": by_source,
        "figure": str(figure_path),
        # The headline reads jspace: H3 is a claim about the WORKSPACE, so
        # recruitment in the orthogonal remainder would not support it.
        "verdict": jspace["verdict"],
        "recruitment_delta": jspace["recruitment_delta"],
        # A delta in jspace means little if the same delta shows up in a random
        # subspace of equal rank — that would be a question-driven shift in the
        # residual generally, not recruitment INTO the workspace. Same capacity
        # logic as RQ1's control.
        "exceeds_capacity_control": bool(
            jspace["recruitment_delta"] > by_source["random_subspace"]["recruitment_delta"]
        ),
        "note": (
            "read at the final token of sentence+question; any position inside "
            "the sentence is question-independent under a causal mask. "
            "Decodability, not use; a linear null does not rule out "
            "multiplicative binding codes (proposal, Potential Limitations)"
        ),
    }
    out = Path(config.paths.results) / "e4_recruitment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _report_block(report: ProbeReport) -> dict[str, Any]:
    return {
        "accuracy": report.accuracy,
        "control_accuracy": report.control_accuracy,
        "selectivity": report.selectivity,
        "n_examples": report.n_examples,
        "n_folds": report.n_folds,
        # Read these before the mean (analysis.probes.ProbeReport): with one
        # fold per concept pair a single inverted pair drags the mean below
        # chance, which reads as absence of signal but is a transfer failure.
        "fold_ids": list(report.fold_ids),
        "fold_accuracies": list(report.fold_accuracies),
        "fold_spread": report.fold_spread,
        "inverting_folds": list(report.inverting_folds),
    }


def _verdict(role_accuracy: float, bag_accuracy: float, delta: float) -> str:
    """Which hypothesis this source's numbers support.

    "recruited" also requires the bag condition near chance: a large delta
    between two ABOVE-chance conditions is modulation of an always-present
    signal, a weaker claim.

    Below-chance conditions are named separately and never fall through to a
    presence verdict. A probe landing consistently below chance under
    leave-one-pair-out has learned a rule that ANTI-transfers across concept
    pairs — evidence about how role is encoded, not that it is absent.
    """
    role_at_chance = abs(role_accuracy - 0.5) < _CHANCE_TOLERANCE
    bag_at_chance = abs(bag_accuracy - 0.5) < _CHANCE_TOLERANCE
    role_inverted = role_accuracy < 0.5 - _CHANCE_TOLERANCE
    bag_inverted = bag_accuracy < 0.5 - _CHANCE_TOLERANCE

    # Checked before anything else: a below-chance condition makes the delta
    # between the two conditions uninterpretable, whatever its size.
    if role_inverted and bag_inverted:
        return "anti_transfer_both"
    if role_inverted or bag_inverted:
        return "anti_transfer_one"

    if role_at_chance and bag_at_chance:
        return "absent"
    if delta >= _RECRUITMENT_THRESHOLD and bag_at_chance:
        return "recruited"
    if delta >= _RECRUITMENT_THRESHOLD:
        return "modulated"
    if delta <= -_RECRUITMENT_THRESHOLD:
        return "inverted"
    return "always_on"
