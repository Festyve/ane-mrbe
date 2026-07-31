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


@runtime_checkable
class ProbeActivationSource(Protocol):
    """A model that exposes the RQ1 activation sources (proposal, §4)."""

    def probe_activation(
        self,
        sentence: str,
        entity: str,
        site: InjectionSite,
    ) -> dict[str, Sequence[float]]:
        """No-edit activations at `site` keyed by source: "jspace" (J-space
        coordinates), "orthogonal" (residual component the lens cannot see),
        "residual" (the full residual-stream activation), and
        "random_subspace" (the capacity control — a random subspace of the same
        rank as jspace; see analysis.probes.PROBE_SOURCES). The RQ1 linear
        probe trains on each source separately."""
        ...


@runtime_checkable
class RecruitmentActivationSource(Protocol):
    """A model that can be read WHILE a given question is in context (E4).

    Separate from ProbeActivationSource because the read POSITION differs, and
    the difference is forced by causal attention rather than chosen.

    E4 presents identical stimulus tokens under a role question and a bag
    question and asks whether binding information appears in the workspace only
    when the task needs it (proposal §6 E4, testing H3). The question is
    appended AFTER the sentence, so under a causal mask it cannot influence any
    token inside the sentence — reading at InjectionSite.FINAL_TOKEN or
    ENTITY_TOKEN would return byte-identical activations for both questions and
    the measured recruitment effect would be exactly zero by construction.

    The read is therefore at the final token of the full prompt (sentence +
    probe), the position where the model is actually composing its answer and
    the only one both questions can differ at. There is no `site` parameter
    because no other position can carry the effect.
    """

    def recruitment_activation(
        self,
        sentence: str,
        probe: str,
        entity: str,
    ) -> dict[str, Sequence[float]]:
        """No-edit activations at the final token of `sentence + probe`, keyed
        by the same sources as probe_activation."""
        ...
