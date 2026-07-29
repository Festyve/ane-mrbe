"""What was actually run: stamped into every results JSON.

A results file that records only numbers cannot be audited later. When a run
is shared as a bare JSON blob, "which model, which lens, which layer?" has no
answer, and a config that drifted from the branch it was supposed to match is
indistinguishable from one that did not. Every RQ summary therefore carries a
`provenance` block naming the model, the lens artifact, the layer band, the
edit sparsity, and the commit — enough to reproduce the run or to disqualify
it.

The git lookup is best-effort: a missing/!repo checkout records None rather
than failing a run that is otherwise fine.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from jspace_binding.config import Config

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str | None:
    """Best-effort `git ...` in the repo root; None if unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def run_provenance(config: Config, n_families: int | None = None) -> dict[str, Any]:
    """Model/lens/layer/commit identity for one run.

    `dirty` flags uncommitted changes: a run made from a dirty tree is not
    reproducible from its commit alone, and silently reporting the commit
    would overstate what the SHA pins down.
    """
    model = config.model
    status = _git("status", "--porcelain")
    return {
        "backend": model.backend,
        "model_id": model.model_id,
        "lens_repo": model.lens_repo,
        "lens_subpath": model.lens_subpath,
        "layer_band": list(model.layer_band) if model.layer_band is not None else None,
        "jspace_k": model.jspace_k,
        "ablate_k": model.ablate_k,
        "dtype": model.dtype,
        "dummy_mode": model.dummy_mode if model.backend == "dummy" else None,
        "n_concept_pairs": len(config.stimuli.concept_pairs),
        "n_families": n_families,
        "seed": config.experiment.seed,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }
