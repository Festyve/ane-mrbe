#!/usr/bin/env python3
"""Export the generated primary stimuli to a reviewable CSV.

Usage: stimuli_to_csv.py [--config configs/default.yaml] [--out PATH]

Why this exists: the primary stimuli live in data/stimuli/*.jsonl, which is
gitignored (regenerable output), so nobody can read the actual sentences and
probes on GitHub. This writes the same content as a CSV that GitHub renders
as a browsable table — the same way the hand-written data/stimuli/<entity>.csv
sheets are readable.

The CSV is a REVIEW ARTIFACT, not an input: the pipeline still reads the
JSONL. Generation is deterministic, so re-running this after any template or
config change reproduces it exactly. Layout mirrors the team's sheets — one
row per item family, the four design cells as columns:

    agent+first | agent+second | patient+first | patient+second

followed by the probes asked after each sentence. recipient_probe is empty for
every construction except the dative (see stimuli.templates).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from jspace_binding.config import Config
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.types import Position, Role

_COLUMNS = [
    "construction",
    "pair_id",
    "entity",
    "counterpart",
    "other_entity",
    "verb",
    "agent_first",
    "agent_second",
    "patient_first",
    "patient_second",
    "role_probe",
    "recipient_probe",
    "neutral_probe",
    # RQ2's recall control, counterbalanced: one asking per participant. Both
    # are needed — scoring either alone reintroduces the base-rate and primacy
    # confounds the pair is there to cancel (see ProbeKind.CONCEPT).
    "concept_probe_entity",
    "concept_probe_other",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/stimuli/primary_stimuli.csv"))
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    families = generate_families(config)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_COLUMNS)
        for family in families:
            writer.writerow(
                [
                    family.construction.value,
                    family.concept_pair.pair_id,
                    family.concept_pair.entity,
                    family.concept_pair.counterpart,
                    family.other_entity,
                    family.verb_lemma,
                    family.cell(Role.AGENT, Position.FIRST).sentence,
                    family.cell(Role.AGENT, Position.SECOND).sentence,
                    family.cell(Role.PATIENT, Position.FIRST).sentence,
                    family.cell(Role.PATIENT, Position.SECOND).sentence,
                    family.role_probe,
                    family.recipient_probe,
                    family.neutral_probe,
                    family.concept_probe_entity,
                    family.concept_probe_other,
                ]
            )

    n_recipient = sum(1 for f in families if f.recipient_probe)
    n_concept = sum(1 for f in families if f.concept_probe_entity and f.concept_probe_other)
    print(f"wrote {len(families)} families -> {args.out}")
    print(f"  {n_recipient} carry a recipient probe (dative only)")
    print(f"  {n_concept} carry both CONCEPT probes (RQ2 recall control)")


if __name__ == "__main__":
    main()
