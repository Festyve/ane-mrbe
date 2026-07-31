"""Backend construction and CLI scaffolding shared by every script.

One place owns the config-to-constructor wiring (previously five scripts each
hand-copied a _build_model and three of them silently dropped
directions.variant) and the not-ready-vs-crashed distinction: preflight()
surfaces missing config decisions / lens artifacts / unfitted directions
BEFORE any sweep starts, so scripts no longer need a broad try/except around
the run itself — a mid-sweep RuntimeError (e.g. CUDA OOM) stays a real error
instead of being mislabeled "backend cannot run yet".
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from jspace_binding.config import Config
from jspace_binding.model.base import WorkspaceModel
from jspace_binding.model.dummy import MODES, DummyModel
from jspace_binding.types import InjectionSite


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    """The three flags every entry point shares."""
    from pathlib import Path

    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--dry-run", action="store_true", help="force the GPU-free DummyModel backend"
    )
    parser.add_argument(
        "--dummy-mode",
        choices=MODES,
        default=None,
        help="DummyModel ground truth (default: config model.dummy_mode)",
    )


def build_model(
    config: Config, dry_run: bool = False, dummy_mode: str | None = None
) -> WorkspaceModel:
    """The one config-to-backend constructor. Always forwards the full wiring
    (directions dir + variant), so no entry point can silently ignore a
    config knob."""
    backend = "dummy" if dry_run else config.model.backend
    if backend == "dummy":
        return DummyModel(mode=dummy_mode or config.model.dummy_mode, seed=config.experiment.seed)
    if backend == "qwen_jlens":
        # Imported lazily: the only backend that needs the heavy `model` extras.
        from jspace_binding.model.qwen_jlens import QwenJLensModel

        return QwenJLensModel(
            config.model,
            directions_dir=config.paths.directions,
            direction_variant=config.directions.variant,
        )
    sys.exit(f"unknown model.backend {config.model.backend!r}; expected 'dummy' or 'qwen_jlens'")


def preflight_or_exit(model: WorkspaceModel, sites: Sequence[InjectionSite] = ()) -> None:
    """Run the backend's preflight (if it has one) and exit 2 with the
    one-shot everything-missing message when it is not runnable yet. Sweeps
    run OUTSIDE this guard so their genuine failures surface as failures."""
    check = getattr(model, "preflight", None)
    if check is None:
        return
    try:
        check(sites)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"backend cannot run yet: {exc}", file=sys.stderr)
        sys.exit(2)
