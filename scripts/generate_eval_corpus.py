#!/usr/bin/env python3
"""Generate the four-construction held-out EVALUATION corpus.

The hand-written eval set (data/handwritten/eval_doctor.jsonl, via
scripts/csv_to_jsonl.py) covers active/passive only, but the primary
experiment pushes one fitted direction across all four constructions. This
emits an eval corpus spanning every construction, disjoint from both the
fitting corpus and the primary set at the sentence, frame_id, and template
level -- the three axes scripts/direction_sanity.py enforces.

Usage:
    generate_eval_corpus.py --entity doctor
    generate_eval_corpus.py --entity doctor,nurse --out-dir data/handwritten
    generate_eval_corpus.py --entity doctor --csv data/stimuli/eval_4construction.csv

--out defaults to data/handwritten/eval_{entity}_4construction.jsonl (one file
per entity). --csv additionally writes a readable export, the same
JSONL-is-the-source / CSV-is-a-view relationship scripts/stimuli_to_csv.py
established for the primary set -- editing the CSV changes nothing.

After generating, run scripts/audit_disjointness.py to confirm the result is
clean against every fitting corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from jspace_binding.stimuli.eval_corpus import generate_eval_corpus
from jspace_binding.stimuli.fitting_corpus import save_fitting_corpus

_CSV_COLUMNS = (
    "entity",
    "construction_frame",
    "role",
    "position",
    "verb",
    "other",
    "sentence",
    "role_probe",
)


def _write_csv(examples, path: Path) -> None:
    """Readable export. Derived from the JSONL, never a source of truth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_COLUMNS)
        for ex in examples:
            writer.writerow(
                [
                    ex.entity,
                    ex.frame_id,
                    ex.role.value,
                    ex.position.value,
                    ex.verb,
                    ex.other,
                    ex.sentence,
                    ex.role_probe,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the four-construction held-out evaluation corpus."
    )
    parser.add_argument(
        "--entity",
        type=lambda s: tuple(e.strip() for e in s.split(",") if e.strip()),
        default=("doctor",),
        help="comma-separated entities to generate (default: doctor)",
    )
    parser.add_argument(
        "--other",
        default="lawyer",
        help="distractor participant appearing opposite the entity (default: lawyer)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/handwritten"),
        help="output directory for the JSONL (default: data/handwritten)",
    )
    parser.add_argument(
        "--verbs-per-cell",
        type=int,
        default=None,
        help="verbs per (construction, cell); default is the full pool (8)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="also write a readable CSV export of every generated entity",
    )
    args = parser.parse_args()

    written: list[dict[str, object]] = []
    combined = []
    for entity in args.entity:
        try:
            examples = generate_eval_corpus(
                (entity,), other_entity=args.other, verbs_per_cell=args.verbs_per_cell
            )
        except ValueError as exc:
            sys.exit(str(exc))
        out = args.out_dir / f"eval_{entity}_4construction.jsonl"
        save_fitting_corpus(examples, out)
        combined.extend(examples)
        frames = sorted({ex.frame_id for ex in examples})
        print(f"wrote {out} ({len(examples)} examples, {len(frames)} frames)", file=sys.stderr)
        written.append(
            {"entity": entity, "path": str(out), "n": len(examples), "frames": frames}
        )

    if args.csv is not None:
        _write_csv(combined, args.csv)
        print(f"wrote {args.csv} ({len(combined)} rows)", file=sys.stderr)

    print(
        json.dumps(
            {
                "other_entity": args.other,
                "csv": None if args.csv is None else str(args.csv),
                "written": written,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
