#!/usr/bin/env python3
"""Convert a hand-written corpus CSV (data/stimuli/<entity>.csv) to the
fitting + eval JSONL the pipeline reads.

Usage: csv_to_jsonl.py --entity doctor
       csv_to_jsonl.py --csv data/stimuli/doctor.csv --out-dir data/handwritten

This makes the CSV the single source of truth: humans edit the spreadsheet,
this regenerates the JSONL the code consumes, and the two can't drift. The
CSV layout (per the team's sheet) is eight columns:

    fitting: agent-active | agent-passive | patient-active | patient-passive
    eval:    agent-active | agent-passive | patient-active | patient-passive

(the 2nd/3rd fitting headers read "AGENT PASSIVE" in the sheet, but the 3rd
column's content is patient-active — we map by position, not by header). The
(verb, other) for each row is read from the agent-active cell and all four
forms are regenerated from templates, so a typo in one cell can't slip
through. An eval form is emitted only where its cell is non-empty (the sheet
leaves some passive eval cells blank).

After converting, run scripts/check_corpus.py to confirm the result is clean
against the primary stimulus set.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from jspace_binding.stimuli.fitting_corpus import FittingExample, save_fitting_corpus
from jspace_binding.types import Position, Role

_AGENT_ACTIVE = re.compile(r"^\s*The\s+(\w+)\s+(\w+)\s+the\s+(\w+)\.\s*$")


def _record(entity: str, verb: str, other: str, role: Role, position: Position) -> FittingExample:
    active = (role is Role.AGENT) == (position is Position.FIRST)
    if role is Role.AGENT and position is Position.FIRST:
        sentence = f"The {entity} {verb} the {other}."
    elif role is Role.AGENT:  # second
        sentence = f"The {other} was {verb} by the {entity}."
    elif position is Position.SECOND:  # patient, active
        sentence = f"The {other} {verb} the {entity}."
    else:  # patient, first — passive
        sentence = f"The {entity} was {verb} by the {other}."
    return FittingExample(
        entity=entity,
        role=role,
        position=position,
        frame_id=f"handwritten_{'active' if active else 'passive'}",
        verb=verb,
        other=other,
        sentence=sentence,
        role_probe=f"Question: Who {verb} someone? Answer: The",
    )


# Column index -> (role, position) for the four fitting and four eval forms.
_FORMS = (
    (Role.AGENT, Position.FIRST),
    (Role.AGENT, Position.SECOND),
    (Role.PATIENT, Position.SECOND),
    (Role.PATIENT, Position.FIRST),
)


def _parse(agent_active_cell: str, entity: str) -> tuple[str, str]:
    match = _AGENT_ACTIVE.match(agent_active_cell)
    if match is None:
        raise ValueError(f"cannot parse agent-active cell {agent_active_cell!r}")
    parsed_entity, verb, other = match.group(1), match.group(2), match.group(3)
    if parsed_entity != entity:
        raise ValueError(
            f"cell subject {parsed_entity!r} != expected entity {entity!r} in "
            f"{agent_active_cell!r}"
        )
    return verb, other


def convert(csv_path: Path, entity: str) -> tuple[list[FittingExample], list[FittingExample]]:
    fitting: list[FittingExample] = []
    eval_set: list[FittingExample] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:  # skip header
        cells = [c.strip() for c in row]
        if len(cells) < 4 or not cells[0]:
            continue  # blank trailing row
        f_verb, f_other = _parse(cells[0], entity)
        for (role, position) in _FORMS:
            fitting.append(_record(entity, f_verb, f_other, role, position))
        if len(cells) >= 8 and cells[4]:
            e_verb, e_other = _parse(cells[4], entity)
            for col, (role, position) in zip(range(4, 8), _FORMS, strict=True):
                if cells[col]:  # eval leaves some passive cells blank
                    eval_set.append(_record(entity, e_verb, e_other, role, position))
    return fitting, eval_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a corpus CSV to fitting/eval JSONL.")
    parser.add_argument("--entity", default=None, help="e.g. doctor (implies default paths)")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/handwritten"))
    args = parser.parse_args()

    if args.entity is None and args.csv is None:
        sys.exit("pass --entity NAME or --csv PATH")
    entity = args.entity or args.csv.stem
    csv_path = args.csv or Path("data/stimuli") / f"{entity}.csv"
    if not csv_path.exists():
        sys.exit(f"no such CSV: {csv_path}")

    fitting, eval_set = convert(csv_path, entity)
    fitting_path = args.out_dir / f"fitting_{entity}.jsonl"
    eval_path = args.out_dir / f"eval_{entity}.jsonl"
    save_fitting_corpus(fitting, fitting_path)
    save_fitting_corpus(eval_set, eval_path)
    print(f"{csv_path} -> {fitting_path} ({len(fitting)} fitting) "
          f"+ {eval_path} ({len(eval_set)} eval)")


if __name__ == "__main__":
    main()
