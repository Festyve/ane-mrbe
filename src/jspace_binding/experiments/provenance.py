"""What was actually run: stamped into every results JSON.

Every summary carries a `provenance` block naming the model, lens artifact,
layer band, edit sparsity, and commit — enough to reproduce a run shared as a
bare JSON blob, or to disqualify it. The git lookup is best-effort.
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


def run_provenance(
    config: Config, n_families: int | None = None, model: object | None = None
) -> dict[str, Any]:
    """Model/lens/layer/commit identity for one run.

    `dirty` flags uncommitted changes, since a run from a dirty tree is not
    reproducible from its commit alone.

    Pass `model` — the backend instance that actually ran. Config alone
    describes what was *requested*, and the two can disagree. When they do, the
    RUN wins and the mismatch is recorded rather than hidden.
    """
    cfg_model = config.model
    status = _git("status", "--porcelain")

    dummy_mode = cfg_model.dummy_mode if cfg_model.backend == "dummy" else None
    mismatch: dict[str, Any] | None = None
    actual_mode = getattr(model, "mode", None)
    if actual_mode is not None and actual_mode != dummy_mode:
        mismatch = {"field": "dummy_mode", "config": dummy_mode, "actual": actual_mode}
        dummy_mode = actual_mode

    return {
        "backend": cfg_model.backend,
        "model_id": cfg_model.model_id,
        "lens_repo": cfg_model.lens_repo,
        "lens_subpath": cfg_model.lens_subpath,
        "layer_band": (
            list(cfg_model.layer_band) if cfg_model.layer_band is not None else None
        ),
        "jspace_k": cfg_model.jspace_k,
        "ablate_k": cfg_model.ablate_k,
        "dtype": cfg_model.dtype,
        "dummy_mode": dummy_mode,
        # Non-null means the running backend disagreed with the config it was
        # supposedly built from. Never silently reconciled — a run whose config
        # does not describe it is not reproducible from that config.
        "config_model_mismatch": mismatch,
        "n_concept_pairs": len(config.stimuli.concept_pairs),
        "n_families": n_families,
        "seed": config.experiment.seed,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }
