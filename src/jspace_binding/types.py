"""Core types shared across the pipeline.

Everything downstream (stimuli, interventions, model backends, analysis) speaks
in these types. Keep this module dependency-free (stdlib only) so any module can
import it without pulling in numpy/torch.

Design note: the unit of analysis is the ItemFamily — the matched quadruple of
sentences spanning all four (role x position) cells for one lexical content in
one construction. The binding score is computed per family, and bootstrap
resampling happens at the family level to preserve the difference-in-differences
pairing.

Primary vs control interventions (proposal, Methods): the primary causal test
is the ROLE_PUSH — a byte-identical push along an in-house-fitted role axis
r_entity, run in both signs as separate uniform conditions. The lexical
identity swap (IDENTITY_SWAP, e.g. doctor->nurse) is retained only as the
intervention-strength control, paired with the NEUTRAL probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    """Thematic role of the target entity in the sentence. The tested variable."""

    AGENT = "agent"
    PATIENT = "patient"


class Position(str, Enum):
    """Linear position of the target entity in the surface sentence.

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


class PushSign(str, Enum):
    """Direction of a role-axis push. Both signs run as separate *uniform*
    conditions — the sign is never chosen by looking at the sentence's own role
    label (a conditional flip could fake the effect; proposal, Key Ideas)."""

    TOWARD_AGENT = "toward_agent"
    TOWARD_PATIENT = "toward_patient"


class EditType(str, Enum):
    """Which intervention (if any) is applied to the residual stream.

    Note: the proposal's "null-edit (neutral probe)" control is NOT a separate
    edit vector — it is IDENTITY_SWAP paired with ProbeKind.NEUTRAL. It varies
    the probe, not the vector, so it does not appear here.
    """

    ROLE_PUSH = "role_push"  # PRIMARY: +/- coefficient * r_entity at the site
    NO_EDIT = "no_edit"  # baseline: model runs untouched
    NULL_NON_PARTICIPANT = "null_non_participant"  # push r of an entity absent from the sentence
    RANDOM_DIRECTION = "random_direction"  # matched-norm random direction, same site
    SHUFFLED_LABEL_DIRECTION = "shuffled_label_direction"  # r refit with shuffled role labels
    IDENTITY_SWAP = "identity_swap"  # CONTROL ONLY: lexical coordinate swap (strength check)
    # RQ2 ablations (experiments.rq2_ablation; never part of the primary sweep):
    ABLATE_JSPACE = "ablate_jspace"  # remove the J-space component at the site
    ABLATE_RANDOM_SUBSPACE = "ablate_random_subspace"  # remove a matched-dim random subspace


# Edit types that push a direction and therefore run in both PushSigns.
DIRECTION_PUSH_EDIT_TYPES: tuple[EditType, ...] = (
    EditType.ROLE_PUSH,
    EditType.NULL_NON_PARTICIPANT,
    EditType.RANDOM_DIRECTION,
    EditType.SHUFFLED_LABEL_DIRECTION,
)


class ProbeKind(str, Enum):
    """Which question is appended after the sentence."""

    ROLE = "role"  # role-diagnostic: the answer depends on who is agent/patient
    NEUTRAL = "neutral"  # role-blind: only checks the edit propagated at all
    # DATIVE only: the ROLE probe there asks for the GIVER (mapped to AGENT),
    # so the recipient — arguably the dative's more interesting participant —
    # is never queried. This probe asks for the recipient instead.
    # ORIENTATION: P(entity) is high when the entity is the RECIPIENT, i.e. the
    # PATIENT slot, so the agent-patient gap and every push inverts relative to
    # the ROLE probe. analysis.binding_score negates it (see PROBE_ORIENTATION)
    # so "positive = binding" still holds. Families whose construction has no
    # recipient probe (ItemFamily.recipient_probe == "") skip it entirely.
    RECIPIENT = "recipient"


class InjectionSite(str, Enum):
    """Token position(s) at which the edit is applied. Role directions are
    fitted separately per site (proposal, Methods)."""

    FINAL_TOKEN = "final_token"  # primary: workspace holds the assembled scene
    ENTITY_TOKEN = "entity_token"  # secondary: workspace is token-local


@dataclass(frozen=True)
class ConceptPair:
    """The target entity plus its identity-swap counterpart.

    `entity` is the concept whose fitted role axis the primary ROLE_PUSH
    intervenes on and whose answer probability the ROLE probe reads (it appears
    in every stimulus sentence). `counterpart` is used ONLY by the IDENTITY_SWAP
    intervention-strength control (e.g. doctor->nurse) and never appears in any
    sentence.
    """

    entity: str
    counterpart: str

    @property
    def pair_id(self) -> str:
        return f"{self.entity}->{self.counterpart}"


@dataclass(frozen=True)
class AnswerSet:
    """Candidate single-token answers whose probabilities we read.

    All three must be single tokens under the target model's tokenizer
    (validated in stimuli.vocab once the tokenizer is available). The ROLE-probe
    readout is P(entity) — pushing the entity's role axis should change whether
    the entity is named as the agent. The NEUTRAL-probe strength check reads
    P(counterpart) rising and P(entity) falling under IDENTITY_SWAP.
    """

    entity: str  # the pushed entity, e.g. "doctor"
    counterpart: str  # identity-swap target, e.g. "nurse" (control only)
    other: str  # the non-target participant, e.g. "lawyer"

    @property
    def tokens(self) -> tuple[str, str, str]:
        return (self.entity, self.counterpart, self.other)


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
    (and therefore the role/position of the target entity) differs.
    """

    family_id: str
    concept_pair: ConceptPair
    construction: Construction
    other_entity: str
    verb_lemma: str
    cells: dict[str, Stimulus] = field(default_factory=dict)  # key: cell_key(role, position)
    role_probe: str = ""
    neutral_probe: str = ""
    # DATIVE only; "" for constructions with no recipient reading. See
    # ProbeKind.RECIPIENT for the orientation caveat.
    recipient_probe: str = ""
    answer_set: AnswerSet | None = None

    def cell(self, role: Role, position: Position) -> Stimulus:
        return self.cells[cell_key(role, position)]


def cell_key(role: Role, position: Position) -> str:
    """Stable string key for a design cell (JSON-serializable)."""
    return f"{role.value}:{position.value}"


@dataclass(frozen=True)
class EditSpec:
    """Backend-agnostic description of one intervention.

    Model backends execute this (interventions.edits documents the math);
    analysis code only reads edit_type and sign.

    Field usage by edit type:
    - ROLE_PUSH / SHUFFLED_LABEL_DIRECTION: entity + sign + coefficient
      (entity names whose fitted direction to push; the shuffled variant loads
      the shuffled-label refit of the same entity's direction).
    - NULL_NON_PARTICIPANT: entity (an entity ABSENT from the sentence) + sign
      + coefficient.
    - RANDOM_DIRECTION: sign + coefficient + seed (matched-norm random unit
      direction in the same subspace).
    - IDENTITY_SWAP: swap_source + swap_target + alpha (lexical coordinate
      swap; the intervention-strength control).
    - NO_EDIT: everything None.
    """

    edit_type: EditType
    entity: str | None = None  # whose fitted role direction (push edits)
    sign: PushSign | None = None  # push direction; None for NO_EDIT / IDENTITY_SWAP
    coefficient: float | None = None  # push scaling; None = backend default
    swap_source: str | None = None  # IDENTITY_SWAP only
    swap_target: str | None = None  # IDENTITY_SWAP only
    alpha: float | None = None  # IDENTITY_SWAP scaling; None = pure swap
    seed: int | None = None  # RNG seed for RANDOM_DIRECTION reproducibility


@dataclass(frozen=True)
class TrialResult:
    """One forward pass: (family cell) x (edit x sign) x (probe) x (site) -> answer probs."""

    family_id: str
    pair_id: str
    construction: Construction
    role: Role
    position: Position
    edit_type: EditType
    probe_kind: ProbeKind
    injection_site: InjectionSite
    answer_probs: dict[str, float]  # answer token -> probability
    push_sign: PushSign | None = None  # set iff edit_type is a direction push
