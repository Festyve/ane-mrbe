"""Model backend protocol.

Every backend — the real Qwen3.6-27B + J-lens stack and the GPU-free
DummyModel — implements this one method. The experiment runner is written
against the protocol only, so the full pipeline can be exercised end-to-end
without a GPU (scripts/run_primary.py --dry-run).
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
        candidates), so "the edit moved mass to nurse" is measured absolutely.
        """
        ...
