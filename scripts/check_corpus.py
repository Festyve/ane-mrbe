#!/usr/bin/env python3
"""QC a fitting/eval corpus JSONL: structure, primary-set collisions, leakage.

Usage:
  check_corpus.py --corpus data/handwritten/fitting_doctor.jsonl \
      [--against data/handwritten/eval_doctor.jsonl] [--config configs/default.yaml]

Checks (stimuli.qc):
- structure: counts, role x position balance, missing roles, duplicates,
  vocabulary hygiene;
- collisions: exact sentences shared with the primary stimulus set generated
  at the config's scale — fitting and testing on the same sentences would
  contaminate the causal test, so any hit is a failure;
- leakage (--against): sentences shared with a second corpus file (e.g.
  fitting vs eval).

Exits 0 when clean, 1 on any collision / leak / duplicate / vocabulary
violation / missing role. Position IMBALANCE is a warning by default —
evaluation sets (e.g. the 8/7/7/8 eval grid) are legitimately unbalanced —
and a hard failure only under --require-balance, which fitting corpora
should use. The JSON report goes to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jspace_binding.config import Config
from jspace_binding.stimuli.fitting_corpus import load_fitting_corpus
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.stimuli.qc import (
    find_cross_leaks,
    find_primary_collisions,
    structure_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="QC a corpus JSONL.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--against", type=Path, default=None, help="second corpus to check for leakage"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--require-balance",
        action="store_true",
        help="treat role x position imbalance as a failure (use for fitting corpora)",
    )
    args = parser.parse_args()

    corpus = load_fitting_corpus(args.corpus)
    families = generate_families(Config.from_yaml(args.config))
    report: dict[str, object] = {
        "corpus": str(args.corpus),
        "structure": structure_report(corpus),
        "primary_collisions": find_primary_collisions(corpus, families),
    }
    if args.against is not None:
        report["leaks_vs"] = str(args.against)
        report["leaks"] = find_cross_leaks(corpus, load_fitting_corpus(args.against))

    print(json.dumps(report, indent=2))
    structure = report["structure"]
    problems = []
    if report["primary_collisions"]:
        problems.append(f"{len(report['primary_collisions'])} primary-set collisions")
    if report.get("leaks"):
        problems.append(f"{len(report['leaks'])} sentences leaked vs {args.against}")
    if structure["duplicate_sentences"]:
        problems.append(f"{len(structure['duplicate_sentences'])} duplicate sentences")
    if structure["unknown_entities"]:
        problems.append(f"unknown entities {structure['unknown_entities']}")
    if structure["missing_roles"]:
        problems.append(f"missing roles {structure['missing_roles']}")
    if not structure["position_balanced"]:
        if args.require_balance:
            problems.append("role x position imbalance (--require-balance)")
        else:
            print("note: role x position counts are unbalanced (ok for eval sets; "
                  "use --require-balance for fitting corpora)", file=sys.stderr)
    if problems:
        print("FAIL: " + "; ".join(problems), file=sys.stderr)
        sys.exit(1)
    print("clean", file=sys.stderr)


if __name__ == "__main__":
    main()
