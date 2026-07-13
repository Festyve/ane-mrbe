"""Qwen3.6-27B + pre-fitted Jacobian-lens backend (stub).

Blocked on open Methods decisions; every entry point fails loudly with a
pointer to what is missing instead of guessing. The class is shaped so filling
it in is mechanical: _load_model -> _load_lens -> _concept_direction ->
_apply_edit_hooks -> answer_distribution.

No torch/transformers imports here — the stub must be importable on GPU-free
machines (install the `model` extra when implementing).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from jspace_binding.config import ModelConfig
from jspace_binding.types import EditSpec, InjectionSite

_RECIPE = (
    'concept directions via "Tell me about {concept}" — read the lens-space state at the '
    "concept token, mean-subtracted over a 100-concept baseline; edits apply the "
    "coordinate swap documented in interventions.edits (V = [v_s v_t], c = V+ h, "
    "h_patched = h + V(sigma(c) - c), optionally scaled by alpha)"
)


class QwenJLensModel:
    """WorkspaceModel backend for the real model.

    The constructor is deliberately non-raising: it validates the config and
    collects the still-open decisions into `missing_decisions`, so the
    runner can construct the backend and report everything that is missing in
    one shot when `answer_distribution` is first called.
    """

    def __init__(self, config: ModelConfig) -> None:
        if config.backend != "qwen_jlens":
            raise ValueError(f"config.backend is {config.backend!r}, expected 'qwen_jlens'")
        self.config = config
        self.missing_decisions: list[str] = []
        if config.layer_band is None:
            self.missing_decisions.append(
                "layer_band: workspace layer band for Qwen3.6-27B (model.layer_band in config)"
            )
        if config.alpha is None:
            self.missing_decisions.append(
                "alpha: coordinate-swap scaling (model.alpha in config)"
            )

    def answer_distribution(
        self,
        sentence: str,
        probe: str,
        edit: EditSpec,
        site: InjectionSite,
        answer_tokens: Sequence[str],
    ) -> dict[str, float]:
        """Intended mechanics: within _apply_edit_hooks(edit, site), run one
        forward pass on `sentence + probe` and read each answer token's
        probability from the full next-token softmax (no renormalization over
        the candidates)."""
        missing = "; ".join(self.missing_decisions) or "none — decisions pinned, code pending"
        raise NotImplementedError(
            f"QwenJLensModel is a stub. Open decisions: {missing}. "
            f"Recipe: {_RECIPE}."
        )

    def _load_model(self) -> None:
        """Load config.model_id at config.dtype; keep model + tokenizer on self.
        Validate answer-token single-tokenness here via stimuli.vocab."""
        raise NotImplementedError("stub: load Qwen3.6-27B once model_id is verified")

    def _load_lens(self) -> None:
        """Load the pre-fitted Jacobian lens from config.lens_repo covering
        config.layer_band."""
        raise NotImplementedError("stub: blocked on layer_band")

    def _concept_direction(self, word: str) -> Any:
        """Lens-space direction for `word`: run "Tell me about {word}", read
        the lens coordinates at the concept token, subtract the mean over the
        100-concept baseline set. Returns a torch.Tensor once implemented
        (typed Any to keep this module import-light)."""
        raise NotImplementedError("stub: baseline concept list + readout token undecided")

    def _apply_edit_hooks(self, edit: EditSpec, site: InjectionSite) -> Any:
        """Context manager installing forward hooks over the layer band that
        apply `edit` at `site`: REAL / NULL_NON_PARTICIPANT use the coordinate
        swap (interventions.edits docstring), RANDOM_DIRECTION a matched-norm
        vector seeded from edit.seed, NO_EDIT installs nothing."""
        raise NotImplementedError("stub: blocked on layer_band and alpha")
