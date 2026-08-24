#!/usr/bin/env python3
"""Copy a run's outputs into the tracked `runs/` tree and push them.

Written for ephemeral GPU boxes. The failure mode it prevents is silent:
.gitignore excludes every output location as regenerable, so on a box that is
about to be deleted `git add -A` stages nothing, reports success, and pushes an
empty commit. `runs/` is deliberately not ignored.

    python scripts/save_run.py rq1-pilot          # copy + commit + push
    python scripts/save_run.py rq2-full --no-push  # commit only

Big trial logs are gzipped; anything still over the GitHub limit is reported
and skipped rather than silently failing the push.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

# Everything worth more than the compute that produced it. Defaults match
# configs/default.yaml; pass --config to read them from a different one.
DEFAULT_SOURCES = (
    Path("data/results"),
    Path("data/directions"),
    Path("figures"),
    Path("data/calibration.json"),
)


def _sources_for(config_path: Path | None) -> tuple[Path, ...]:
    """Output locations to archive, read from the config that produced them.

    Non-default configs write to their own paths, so hard-coded defaults would
    silently archive an earlier run's files under this run's name.
    """
    if config_path is None:
        return DEFAULT_SOURCES
    from jspace_binding.config import Config

    paths = Config.from_yaml(config_path).paths
    return (
        Path(paths.results),
        Path(paths.directions),
        Path(paths.figures),
        Path(paths.calibration),
    )

# GitHub hard-rejects blobs over 100 MB; stay clear of the warning band too.
_GZIP_OVER = 5 * 1024 * 1024
_SKIP_OVER = 90 * 1024 * 1024


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git, surfacing its own error text on failure.

    CalledProcessError reports only the command line and an exit code, throwing
    away git's explanation — usually an unset user.name/email on a fresh box.
    Failing to save results on a
    machine that deletes itself is the worst place to hide an error message.
    """
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        hint = ""
        if "tell me who you are" in detail or "user.email" in detail:
            hint = (
                "\n\nFIX: git identity is unset on this machine.\n"
                '  git config --global user.email "you@example.com"\n'
                '  git config --global user.name "Your Name"\n'
                "then re-run this script."
            )
        raise SystemExit(f"git {' '.join(args)} failed:\n{detail}{hint}")
    return result


def _copy(src: Path, dest: Path) -> tuple[int, list[str]]:
    """Copy src -> dest, gzipping large files. Returns (n_copied, skipped)."""
    copied, skipped = 0, []
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    for path in files:
        relative = Path(path.name) if src.is_file() else path.relative_to(src)
        size = path.stat().st_size
        if size > _SKIP_OVER:
            skipped.append(f"{path.relative_to(REPO)} ({size / 1e6:.0f} MB, over limit)")
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if size > _GZIP_OVER:
            with path.open("rb") as fh, gzip.open(f"{target}.gz", "wb") as out:
                shutil.copyfileobj(fh, out)
        else:
            shutil.copy2(path, target)
        copied += 1
    return copied, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="run name, e.g. 'rq1-pilot' or 'rq2-full'")
    parser.add_argument("--no-push", action="store_true", help="commit but do not push")
    parser.add_argument("--note", default="", help="one line recorded alongside the outputs")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config the run used; its paths decide what gets archived. REQUIRED "
             "for any config with non-default output paths (e.g. "
             "configs/expanded_pairs.yaml), otherwise the previous run's files "
             "are archived under this run's name",
    )
    args = parser.parse_args()
    sources = _sources_for(args.config)
    if args.config:
        print(f"archiving paths from {args.config}")

    destination = RUNS / args.label
    destination.mkdir(parents=True, exist_ok=True)

    total, all_skipped, missing = 0, [], []
    for source in sources:
        absolute = REPO / source
        if not absolute.exists():
            missing.append(str(source))
            continue
        copied, skipped = _copy(absolute, destination / source.name)
        total += copied
        all_skipped.extend(skipped)
        print(f"  {source} -> {copied} file(s)")

    if missing:
        print(f"  (absent, not yet produced: {', '.join(missing)})")
    if all_skipped:
        print("\n  SKIPPED, too large for GitHub — move these to Drive/HF by hand:")
        for item in all_skipped:
            print(f"    {item}")

    if total == 0:
        print("\nNothing copied. Run an experiment first.", file=sys.stderr)
        raise SystemExit(1)

    # Provenance the outputs cannot carry themselves: which commit produced them.
    commit = _git("rev-parse", "HEAD", check=False).stdout.strip()
    dirty = bool(_git("status", "--porcelain", check=False).stdout.strip())
    (destination / "RUN_INFO.txt").write_text(
        f"label: {args.label}\ncommit: {commit}\ndirty: {dirty}\n"
        f"config: {args.config or 'configs/default.yaml (assumed)'}\n"
        f"sources: {[str(x) for x in sources]}\n"
        f"note: {args.note}\nfiles: {total}\n",
        encoding="utf-8",
    )

    _git("add", "runs")
    staged = _git("diff", "--cached", "--name-only", check=False).stdout.strip()
    if not staged:
        print("\nNo change vs what is already saved — nothing to commit.")
        return
    _git("commit", "-m", f"Save run outputs: {args.label}\n\n{args.note}".strip())
    print(f"\ncommitted {total} file(s) under runs/{args.label}")

    if args.no_push:
        print("--no-push set; remember these are NOT safe until pushed.")
        return
    pushed = _git("push", check=False)
    if pushed.returncode == 0:
        print("pushed — outputs are now off this machine.")
    else:
        print(f"PUSH FAILED:\n{pushed.stderr}", file=sys.stderr)
        print("Outputs are still LOCAL ONLY. Resolve and push before shutdown.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
