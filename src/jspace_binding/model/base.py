"""Model backend protocols.

Every backend implements WorkspaceModel, and the runners are written against
the protocol only, so the pipeline runs end-to-end without a GPU. Backends that
also serve direction fitting implement FittingActivationSource.
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


class FittingGradientSource(Protocol):
    """A model that exposes readout gradients for the LRE-style estimator
    (directions.estimator = "lre_gradient"; Chanin et al. 2023)."""

    def fitting_gradient(
        self,
        sentence: str,
        role_probe: str,
        entity: str,
        other: str,
        site: InjectionSite,
    ) -> Sequence[float]:
        """d(z_entity - z_other)/dh at the read layer, `site` token, for the
        sentence's role probe — the local steering direction of the role
        readout, in RAW residual space (unlike fitting_activation's J-space
        coordinates: the gradient is what the push should follow, and the push
        is applied in residual space). No edit is applied; the caller
        (directions.fit) pools rows across both roles' exemplars."""
        ...


@runtime_checkable
class ProbeActivationSource(Protocol):
    """A model that exposes the RQ1 activation sources (proposal, §4)."""

    def probe_activation(
        self,
        sentence: str,
        entity: str,
        site: InjectionSite,
        capacity_seed: int | None = None,
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

    Separate from ProbeActivationSource because the read position differs, and
    causal attention forces the difference: the question follows the sentence,
    so reading at FINAL_TOKEN or ENTITY_TOKEN would return byte-identical
    activations for both questions and the effect would be zero by
    construction. The read is at the final token of the full prompt — the only
    position the two questions can differ at, hence no `site` parameter.
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
