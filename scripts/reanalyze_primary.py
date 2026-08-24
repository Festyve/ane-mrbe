#!/usr/bin/env python3
"""Re-score an archived primary run under the CURRENT analysis code.

`run_primary.py` prints its analysis to stdout and writes no summary JSON, so
an archived verdict is frozen at whatever `analyze()` did on the day it ran and
nothing in the log says which side of a later change it fell on. Commit
`6dc4258` gave the neutral strength check a magnitude floor, for instance, and
every log written before it recorded `passes` under a signs-only rule.

The forward passes are already spent, so re-scoring needs only the archived
`trials.jsonl` — no model, GPU, or lens.

    python scripts/reanalyze_primary.py --run runs/gemma3-12b-lre --site entity_token

Exits 1 if the recomputed verdict differs from the archived one, so this can
gate a rerun rather than being read by eye.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jspace_binding.config import Config  # noqa: E402
from jspace_binding.experiments.primary import analyze  # noqa: E402
from jspace_binding.types import (  # noqa: E402
    Construction,
    EditType,
    InjectionSite,
    Position,
    ProbeKind,
    PushSign,
    Role,
    TrialResult,
)


def find_trials(run: Path) -> Path:
    """The run's trials file, in whichever shape save_run.py left it.

    Archives differ in two ways that are not the caller's business: the
    diff-of-means runs write `results/` where the LRE runs write `results_lre/`,
    and anything over save_run's size threshold arrives gzipped.
    """
    candidates = [
        run / results / name
        for results in ("results", "results_lre")
        for name in ("trials.jsonl", "trials.jsonl.gz")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(c.relative_to(run)) for c in candidates)
    raise SystemExit(f"no trials file under {run} (looked for {searched})")


def load_trials(path: Path) -> list[TrialResult]:
    """Inverse of primary._write_trials. Field-for-field, no defaulting: a
    record missing a key is a corrupt archive and should raise, not silently
    analyze a subset."""
    trials = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                sign = r.get("push_sign")
                trials.append(
                    TrialResult(
                        family_id=r["family_id"],
                        pair_id=r["pair_id"],
                        construction=Construction(r["construction"]),
                        role=Role(r["role"]),
                        position=Position(r["position"]),
                        edit_type=EditType(r["edit_type"]),
                        probe_kind=ProbeKind(r["probe_kind"]),
                        injection_site=InjectionSite(r["injection_site"]),
                        answer_probs=r["answer_probs"],
                        push_sign=PushSign(sign) if sign else None,
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not trials:
        raise ValueError(f"{path}: no trials")
    return trials


def archived_verdict(run_dir: Path) -> dict | None:
    """The verdict block from the newest primary log in the archive, if any.

    The logs interleave stderr (weight-loading progress) with the stdout JSON,
    so this pulls the last balanced `"verdict": {...}` object rather than
    trying to parse the file as a whole.
    """
    logs = sorted(run_dir.glob("logs/primary*.log"))
    if not logs:
        return None
    text = logs[-1].read_text(encoding="utf-8", errors="replace")
    start = text.rfind('"verdict"')
    if start == -1:
        return None
    brace = text.find("{", start)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[brace : i + 1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="archived run directory")
    parser.add_argument(
        "--site",
        choices=[s.value for s in InjectionSite],
        default=InjectionSite.ENTITY_TOKEN.value,
        help="site to score; must be one the sweep actually covered",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="override the run's archived config_as_run.yaml (rarely correct)",
    )
    args = parser.parse_args()

    config_path = args.config or args.run / "config_as_run.yaml"
    if not config_path.exists():
        sys.exit(
            f"no config at {config_path}. Pre-{'f2cd63c'} archives predate "
            "config_as_run.yaml; pass --config with the config that run used."
        )
    config = Config.from_yaml(config_path)
    trials_path = find_trials(args.run)
    trials = load_trials(trials_path)
    print(f"loaded {len(trials)} trials from {trials_path}", file=sys.stderr)

    summary = analyze(config, trials, site=InjectionSite(args.site))
    print(json.dumps(summary, indent=2))

    now = summary["verdict"]
    before = archived_verdict(args.run)
    if before is None:
        print("\nno archived verdict to compare against", file=sys.stderr)
        return

    swap = summary["neutral_strength_check"]
    push = summary["push_strength_check"]
    print("\n" + "=" * 68, file=sys.stderr)
    print(f"archived outcome : {before.get('outcome')}", file=sys.stderr)
    print(f"recomputed       : {now['outcome']}", file=sys.stderr)
    print(
        f"strength check   : {before.get('strength_check_passes')} -> "
        f"{now['strength_check_passes']}  (gated on {now.get('strength_gated_on')})",
        file=sys.stderr,
    )
    if push.get("available"):
        strongest = push["strongest_control"]
        print(
            f"  push arm       : displacement {push['role_push_displacement']:.3f} "
            f"vs floor {push['min_push_displacement']}",
            file=sys.stderr,
        )
        print(
            f"                   strongest control {strongest} "
            f"{push['control_displacement'][strongest]:.3f}, "
            f"paired CI {push['vs_control_ci']}",
            file=sys.stderr,
        )
    if swap.get("available"):
        print(
            f"  swap arm       : counterpart {swap['counterpart_shift']:+.5f} "
            f"vs floor {swap.get('min_counterpart_shift')}, "
            f"entity {swap['entity_shift']:+.5f}  "
            f"(reported, not gating)",
            file=sys.stderr,
        )
    print("=" * 68, file=sys.stderr)

    if before.get("outcome") != now["outcome"]:
        print(
            f"\nVERDICT CHANGED: the archived label {before.get('outcome')!r} is "
            f"stale; cite {now['outcome']!r}.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("\nverdict unchanged under current code.", file=sys.stderr)


if __name__ == "__main__":
    main()
