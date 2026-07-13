"""Core types shared across the pipeline.

Everything downstream (stimuli, interventions, model backends, analysis) speaks
in these types. Keep this module dependency-free (stdlib only) so any module can
import it without pulling in numpy/torch.

Design note: the unit of analysis is the ItemFamily — the matched quadruple of
sentences spanning all four (role x position) cells for one lexical content in
one construction. The binding score is computed per family, and bootstrap
resampling happens at the family level to preserve the difference-in-differences
pairing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    """Thematic role of the target concept in the sentence. The tested variable."""

    AGENT = "agent"
    PATIENT = "patient"


class Position(str, Enum):
    """Linear position of the target concept in the surface sentence.

    Nuisance axis: crossed with Role in the 2x2 so word-order effects can be
    averaged out (see analysis.binding_score.position_average).
    """

    FIRST = "first"
    SECOND = "second"


class Construction(str, Enum):
    """Syntactic construction family. Results are reported per construction,
    never pooled-only, so a single-construction effect cannot masquerade as
    general binding."""

    ACTIVE_PASSIVE = "active_passive"
    DATIVE = "dative"
    CLEFT = "cleft"
    RELATIVE_CLAUSE = "relative_clause"


class EditType(str, Enum):
    """Which intervention (if any) is applied to the residual stream.

    Note: the proposal's "null-edit (neutral probe)" control is NOT a separate
    edit vector — it is the REAL edit paired with ProbeKind.NEUTRAL. It varies
    the probe, not the vector, so it does not appear here.
    """

    REAL = "real"  # coordinate-swap source->target (e.g. doctor->nurse)
    NO_EDIT = "no_edit"  # baseline: model runs untouched
    NULL_NON_PARTICIPANT = "null_non_participant"  # swap a concept absent from the sentence
    RANDOM_DIRECTION = "random_direction"  # matched-norm random vector, same site


class ProbeKind(str, Enum):
    """Which question is appended after the sentence."""

    ROLE = "role"  # role-diagnostic: the answer depends on who is agent/patient
    NEUTRAL = "neutral"  # role-blind: only checks the edit propagated at all


class InjectionSite(str, Enum):
    """Token position(s) at which the edit is applied."""

    FINAL_TOKEN = "final_token"  # primary: workspace holds the assembled scene
    ENTITY_TOKEN = "entity_token"  # secondary: workspace is token-local


@dataclass(frozen=True)
class ConceptPair:
    """A source->target concept swap, e.g. doctor->nurse."""

    source: str
    target: str

    @property
    def pair_id(self) -> str:
        return f"{self.source}->{self.target}"


@dataclass(frozen=True)
class AnswerSet:
    """Candidate single-token answers whose probabilities we read.

    All three must be single tokens under the target model's tokenizer
    (validated in stimuli.vocab once the tokenizer is available).
    """

    source: str  # the swapped-out concept, e.g. "doctor"
    target: str  # the swapped-in concept, e.g. "nurse"
    other: str  # the non-target participant, e.g. "lawyer"

    @property
    def tokens(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.other)


@dataclass(frozen=True)
class Stimulus:
    """One sentence: a single cell of the role x position design."""

    role: Role
    position: Position
    sentence: str


@dataclass(frozen=True)
class ItemFamily:
    """The matched quadruple: all four (role x position) cells for one lexical
    content in one construction. Unit of analysis for the binding score.

    Both probes are byte-identical across all four cells — only the sentence
    (and therefore the role/position of the target concept) differs.
    """

    family_id: str
    concept_pair: ConceptPair
    construction: Construction
    other_entity: str
    verb_lemma: str
    cells: dict[str, Stimulus] = field(default_factory=dict)  # key: cell_key(role, position)
    role_probe: str = ""
    neutral_probe: str = ""
    answer_set: AnswerSet | None = None

    def cell(self, role: Role, position: Position) -> Stimulus:
        return self.cells[cell_key(role, position)]


def cell_key(role: Role, position: Position) -> str:
    """Stable string key for a design cell (JSON-serializable)."""
    return f"{role.value}:{position.value}"


@dataclass(frozen=True)
class EditSpec:
    """Backend-agnostic description of one intervention.

    Model backends execute this (interventions.edits documents the
    coordinate-swap math); analysis code only reads edit_type.
    """

    edit_type: EditType
    source_concept: str | None = None  # None for NO_EDIT / RANDOM_DIRECTION
    target_concept: str | None = None
    alpha: float | None = None  # swap scaling; None = backend default. TODO(Monday)
    seed: int | None = None  # RNG seed for RANDOM_DIRECTION reproducibility


@dataclass(frozen=True)
class TrialResult:
    """One forward pass: (family cell) x (edit) x (probe) x (site) -> answer probs."""

    family_id: str
    pair_id: str
    construction: Construction
    role: Role
    position: Position
    edit_type: EditType
    probe_kind: ProbeKind
    injection_site: InjectionSite
    answer_probs: dict[str, float]  # answer token -> probability
