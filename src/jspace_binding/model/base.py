"""Model backend protocols.

Every backend — the real Qwen3.6-27B + J-lens stack and the GPU-free
DummyModel — implements WorkspaceModel. The experiment runner is written
against the protocol only, so the full pipeline can be exercised end-to-end
without a GPU (scripts/run_primary.py --dry-run).

Backends that can also serve the direction-fitting pipeline additionally
implement FittingActivationSource (scripts/fit_directions.py is written
against it).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jspace_binding.types import EditSpec, InjectionSite


@runtime_checkable
class WorkspaceModel(Protocol):
    """A model whose verbalizable workspace we can read and edit."""

    def answer_distribution(
        self,
        sentence: str,
        probe: str,
        edit: EditSpec,
        site: InjectionSite,
        answer_tokens: Sequence[str],
    ) -> dict[str, float]:
        """Run `sentence + probe` with `edit` applied at `site`; return the
        next-token probability for each candidate in `answer_tokens`.

        Probabilities are read from the full softmax (not renormalized over the
        candidates), so "the edit moved mass to the entity" is measured
        absolutely. Analysis converts to log-odds (analysis.binding_score).
        """
        ...


@runtime_checkable
class FittingActivationSource(Protocol):
    """A model that exposes J-space-projected activations for direction fitting."""

    def fitting_activation(
        self,
        sentence: str,
        entity: str,
        site: InjectionSite,
    ) -> Sequence[float]:
        """J-space coordinates of the residual activation for `sentence` at
        `site` (the entity's token, or the sentence-final token). No edit is
        applied; the caller (directions.fit) does the difference-of-means."""
        ...
