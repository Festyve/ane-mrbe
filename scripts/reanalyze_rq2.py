#!/usr/bin/env python3
"""Re-score archived RQ2 ablation verdicts under the CURRENT verdict rule.

`rq2_ablation.json` stores its verdict alongside the deltas that produced it,
so an archived file is frozen at whatever `_involvement_verdict` did on the day
it ran. Two commits have since changed that rule:

    dd6e4fa  two IMPROVEMENTS must not read as causal involvement (sign clause)
    e61aa81  require a MAGNITUDE of binding damage (_MIN_BINDING_DEFICIT)

Neither changed a measurement. Both changed which measurements were allowed to
be called a result, which is the failure mode this project has now hit four
times. The deltas in every archived run remain valid; only the boolean on top
of them moved.

The verdict is a pure function of numbers the archive already holds
(`binding_deficit`, `binding_specific_deficit`, `ci_excludes_zero`), so
re-scoring needs no model, no GPU, no lens, and no rerun:

    python scripts/reanalyze_rq2.py --run runs/gemma3-12b-pilot
    python scripts/reanalyze_rq2.py --all                # scan every run
    python scripts/reanalyze_rq2.py --all --write        # correct in place

Exits 1 if any verdict moves, so this can gate CI or a release rather than
being read by eye. `--write` rewrites the verdict fields in place and records
what it did under `verdict_rescored`, leaving every delta untouched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jspace_binding.experiments.rq2_ablation import (  # noqa: E402
    _MIN_BINDING_DEFICIT,
    _involvement_verdict,
)

_JSPACE = "ablate_jspace"
_RANDOM = "ablate_random_subspace"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rescore_site(site_block: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    """Recompute one site's verdict from its archived deltas.

    Raises on a missing delta rather than defaulting: an archive that cannot
    answer the question should say so, not quietly score a subset.
    """
    deltas = site_block.get("deltas")
    if not deltas or _JSPACE not in deltas:
        raise KeyError(f"no {_JSPACE!r} deltas to re-score")
    jspace = deltas[_JSPACE]
    for key in ("binding_deficit", "binding_specific_deficit", "ci_excludes_zero"):
        if key not in jspace:
            raise KeyError(f"{_JSPACE} is missing {key!r}; archive predates this field")
    random_specific = float(deltas.get(_RANDOM, {}).get("binding_specific_deficit", 0.0))
    verdict = _involvement_verdict(
        float(jspace["binding_deficit"]),
        float(jspace["binding_specific_deficit"]),
        random_specific,
        bool(jspace["ci_excludes_zero"]),
    )
    return verdict, {
        "binding_deficit": float(jspace["binding_deficit"]),
        "recall_deficit": float(jspace.get("recall_deficit", float("nan"))),
        "binding_specific_deficit": float(jspace["binding_specific_deficit"]),
        "random_specific_deficit": random_specific,
    }


def rescore_file(path: Path, write: bool = False) -> bool:
    """Re-score one rq2_ablation.json. Returns True if any verdict moved."""
    report = json.loads(path.read_text(encoding="utf-8"))
    archived_top = report.get("workspace_causally_involved")
    moved = False
    recomputed_top = False
    lines: list[str] = []

    for site, block in report.get("sites", {}).items():
        archived = block.get("workspace_causally_involved")
        verdict, d = rescore_site(block)
        recomputed_top = recomputed_top or verdict
        changed = archived is not None and bool(archived) != verdict
        moved = moved or changed
        # Name WHY a true became false, so the reader does not have to
        # rediscover which clause fired.
        reason = ""
        if archived and not verdict:
            if d["binding_deficit"] <= _MIN_BINDING_DEFICIT:
                reason = (
                    f" [binding_deficit {d['binding_deficit']:+.5f} <= floor "
                    f"{_MIN_BINDING_DEFICIT}: "
                    + (
                        "ablation IMPROVED binding"
                        if d["binding_deficit"] < 0
                        else "binding effectively untouched"
                    )
                    + "]"
                )
            elif d["binding_specific_deficit"] <= d["random_specific_deficit"]:
                reason = " [does not beat the matched random subspace]"
        lines.append(
            f"    {site:<13} {str(archived):<5} -> {str(verdict):<5}"
            f"{'  CHANGED' if changed else ''}"
            f"  binding={d['binding_deficit']:+.5f} recall={d['recall_deficit']:+.5f}"
            f" specific={d['binding_specific_deficit']:+.5f}{reason}"
        )
        if write:
            block["workspace_causally_involved"] = verdict

    top_changed = archived_top is not None and bool(archived_top) != recomputed_top
    moved = moved or top_changed

    print(f"  {path}")
    for line in lines:
        print(line)
    print(
        f"    {'TOP-LEVEL':<13} {str(archived_top):<5} -> {str(recomputed_top):<5}"
        f"{'  CHANGED' if top_changed else ''}"
    )

    if write and moved:
        report["workspace_causally_involved"] = recomputed_top
        # Provenance for the rewrite itself: without it the corrected file is
        # indistinguishable from one that was always right.
        report["verdict_rescored"] = {
            "by": "scripts/reanalyze_rq2.py",
            "at_commit": _git_commit(),
            "min_binding_deficit": _MIN_BINDING_DEFICIT,
            "previous_workspace_causally_involved": archived_top,
            "note": (
                "Verdict only. No delta was recomputed or changed; the "
                "measurements in this file are exactly as originally run."
            ),
        }
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"    rewrote {path}")

    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path, help="one archived run directory")
    group.add_argument("--all", action="store_true", help="scan every run under runs/")
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="root to scan with --all"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="correct verdicts in place (deltas are never touched)",
    )
    args = parser.parse_args()

    if args.all:
        paths = sorted(args.runs_dir.glob("*/results/rq2_ablation.json"))
    else:
        paths = [args.run / "results" / "rq2_ablation.json"]
    if not paths:
        sys.exit("no rq2_ablation.json found")

    print(f"re-scoring {len(paths)} file(s) at _MIN_BINDING_DEFICIT={_MIN_BINDING_DEFICIT}\n")
    stale = [p for p in paths if rescore_file(p, write=args.write)]

    print()
    if not stale:
        print("all archived RQ2 verdicts agree with the current rule.")
        return
    print(f"{len(stale)} file(s) {'corrected' if args.write else 'STALE'}:")
    for p in stale:
        print(f"  {p}")
    if not args.write:
        print("\nre-run with --write to correct them in place.")
    sys.exit(1)


if __name__ == "__main__":
    main()
