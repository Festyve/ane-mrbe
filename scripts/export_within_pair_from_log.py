#!/usr/bin/env python3
"""Recover a within-pair result JSON from the run's stdout log.

    python scripts/export_within_pair_from_log.py \
        runs/gemma3-12b-lre/logs/within_pair_0032.log \
        runs/gemma3-12b-lre/results/within_pair_final_token.json

WHY THIS EXISTS. `scripts/check_within_pair.py` writes its JSON to a path
derived from the config, not from the site, so a second run at a different
injection site overwrites the first. That happened on Gemma-3-12B: the
`final_token` run (log `within_pair_0032.log`) wrote
`results/within_pair.json`, and the `entity_token` run forty minutes later
(log `within_pair_0035.log`) overwrote it. The surviving JSON holds the
entity-token scores, so the final-token scores -- which the paper reports --
existed only inside the log.

PRECISION. The log prints three decimals, but the underlying accuracies are
k/320 for integer k, and at that denominator every three-decimal value maps to
exactly one k. This script reconstructs the exact value and refuses to write
unless (a) each printed value is uniquely recoverable and (b) the mean of the
recovered per-pair values reproduces the printed mean. It never invents
precision it cannot verify.

Exit codes: 0 wrote the file; 1 the log could not be parsed; 2 reconstruction
was ambiguous or failed its self-check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Within-pair splits are 320 scored items per concept pair. Verified against
# the Gemma-3-27B-IT JSON, whose stored values are all exact multiples of 1/320.
DENOMINATOR = 320

_HEADER = re.compile(
    r"model=(?P<model>\S+)\s+site=(?P<site>\w+)\s+families=(?P<families>\d+)"
)
_SOURCE = re.compile(r"^\s{2}(\w+)\s*$")
_PAIR = re.compile(r"^\s{4}(\S+)\s+within-pair\s+([0-9.]+)\s*$")
_MEAN = re.compile(r"^\s{4}MEAN\s+([0-9.]+)\s*$")
_VERDICT = re.compile(r"^ROLE IS PRESENT, BUT NOT FILLER-GENERAL", re.M)


def _exact(printed: str) -> float:
    """Recover the exact k/DENOMINATOR behind a 3-decimal print.

    Raises SystemExit(2) if more or fewer than one k rounds to this value,
    rather than guessing.
    """
    places = len(printed.split(".")[1]) if "." in printed else 0
    target = float(printed)
    hits = [k for k in range(DENOMINATOR + 1) if round(k / DENOMINATOR, places) == target]
    if len(hits) != 1:
        sys.exit(
            f"[2] {printed} maps to {len(hits)} candidates at 1/{DENOMINATOR}; "
            "refusing to guess. Re-run the experiment instead."
        )
    return hits[0] / DENOMINATOR


def parse(text: str) -> dict:
    header = _HEADER.search(text)
    if not header:
        sys.exit("[1] no 'model=... site=... families=...' header found")

    sources: dict[str, dict] = {}
    current: str | None = None
    for line in text.splitlines():
        if (m := _SOURCE.match(line)) and not line.strip().startswith("MEAN"):
            current = m.group(1)
            sources[current] = {}
        elif (m := _PAIR.match(line)) and current:
            sources[current][m.group(1)] = _exact(m.group(2))
        elif (m := _MEAN.match(line)) and current:
            pairs = [v for k, v in sources[current].items() if k != "mean"]
            if not pairs:
                sys.exit(f"[1] MEAN for {current} with no preceding pair rows")
            recovered = sum(pairs) / len(pairs)
            if round(recovered, 3) != float(m.group(1)):
                sys.exit(
                    f"[2] {current}: recovered mean {recovered:.6f} does not "
                    f"reproduce the logged {m.group(1)}"
                )
            sources[current]["mean"] = recovered
            current = None

    if not sources:
        sys.exit("[1] no source blocks parsed")

    return {
        "site": header.group("site"),
        "n_families": int(header.group("families")),
        "within_pair": sources,
        "verdict": (
            "present_not_filler_general"
            if _VERDICT.search(text)
            else "unknown_see_log"
        ),
        "derived_from": {
            "log": None,  # filled in by main()
            "model_id": header.group("model"),
            "method": (
                f"Per-pair accuracies reconstructed as k/{DENOMINATOR} from the "
                "three-decimal values printed in the log; each is uniquely "
                "recoverable at that denominator and the recovered means "
                "reproduce the logged means. Not a re-run."
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--force", action="store_true", help="overwrite an existing out")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(f"[1] {args.out} exists; pass --force to overwrite")

    record = parse(args.log.read_text(encoding="utf-8", errors="replace"))
    record["derived_from"]["log"] = str(args.log)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    wp = record["within_pair"]
    print(f"wrote {args.out}")
    print(f"  site={record['site']}  n_families={record['n_families']}")
    for source, vals in wp.items():
        print(f"  {source:16s} mean={vals['mean']:.6f}")


if __name__ == "__main__":
    main()
