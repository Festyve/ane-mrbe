"""Sentence templates: one matched 2x2 (role x position) quadruple per construction.

The sentence-side "target" is pair.entity — the entity whose fitted role axis
the ROLE_PUSH intervenes on (pair.counterpart never appears in any sentence;
it exists only for the IDENTITY_SWAP strength control). Target and other
entity each appear exactly once per sentence, and only their role and linear
order differ across the four cells, so a role effect that survives
position-averaging cannot be a word-order artifact.

Verbs are stored in simple-past form (identical to the past participle for
every verb used here), so templates need no conjugation logic.
"""

from __future__ import annotations

from jspace_binding.stimuli.vocab import PROFESSION_CUE
from jspace_binding.types import (
    AnswerSet,
    ConceptPair,
    Construction,
    ItemFamily,
    Position,
    Role,
    Stimulus,
    cell_key,
)

# Shared across ACTIVE_PASSIVE / CLEFT / RELATIVE_CLAUSE: person-on-person
# transitives whose past form doubles as the participle.
_TRANSITIVE_VERBS: tuple[str, ...] = (
    "treated",
    "examined",
    "interviewed",
    "photographed",
    "praised",
    "criticized",
    "hired",
    "fired",
    "tutored",
    "sketched",
)

VERBS_BY_CONSTRUCTION: dict[Construction, tuple[str, ...]] = {
    Construction.ACTIVE_PASSIVE: _TRANSITIVE_VERBS,
    Construction.CLEFT: _TRANSITIVE_VERBS,
    Construction.RELATIVE_CLAUSE: _TRANSITIVE_VERBS,
    Construction.DATIVE: ("handed", "sent", "mailed", "offered", "promised"),
}

# Probe drafts, pending team validation of wording. DATIVE needs its own
# frame: "Who handed someone?" is ungrammatical for ditransitives, so the
# dative probe includes the fixed theme from the stimulus sentences.
_ROLE_PROBE_TEMPLATES: dict[Construction, str] = {
    Construction.ACTIVE_PASSIVE: "Question: Who {verb_past} someone? Answer: The",
    Construction.CLEFT: "Question: Who {verb_past} someone? Answer: The",
    Construction.RELATIVE_CLAUSE: "Question: Who {verb_past} someone? Answer: The",
    Construction.DATIVE: "Question: Who {verb_past} a letter to someone? Answer: The",
}
_NEUTRAL_PROBE = "Question: Which professions are mentioned? Answer: The"

# ProbeKind.CONCEPT. Both participants are in the sentence, so the answer turns
# on knowing what the profession IS, not on whether its token appeared. Shared
# by every construction: the question references neither the verb nor the roles,
# which is what makes it role-blind.
_CONCEPT_PROBE_TEMPLATE = "Question: Which one {cue}? Answer: The"

# The ROLE probe above queries the dative's GIVER, so the recipient — arguably
# the dative's more interesting participant — is never asked about. This second
# probe queries the recipient, so the dative can be scored from both sides.
# DATIVE only: the other three constructions have no recipient reading, and
# build_family leaves their recipient_probe empty so the sweep skips them.
# Byte-identical across the four cells, like every other probe.
_RECIPIENT_PROBE_TEMPLATES: dict[Construction, str] = {
    Construction.DATIVE: "Question: Who was {verb_past} the letter by someone? Answer: The",
}

_Cell = tuple[Role, Position]

# Pinned by the proposal's worked example (doctor/lawyer/treated).
_ACTIVE_PASSIVE: dict[_Cell, str] = {
    (Role.AGENT, Position.FIRST): "The {target} {verb_past} the {other}.",
    (Role.AGENT, Position.SECOND): "The {other} was {verb_past} by the {target}.",
    (Role.PATIENT, Position.FIRST): "The {target} was {verb_past} by the {other}.",
    (Role.PATIENT, Position.SECOND): "The {other} {verb_past} the {target}.",
}

# Draft cleft 2x2, pending team validation.
_CLEFT: dict[_Cell, str] = {
    (Role.AGENT, Position.FIRST): "It was the {target} who {verb_past} the {other}.",
    (Role.AGENT, Position.SECOND): "It was the {other} whom the {target} {verb_past}.",
    (Role.PATIENT, Position.FIRST): "It was the {target} whom the {other} {verb_past}.",
    (Role.PATIENT, Position.SECOND): "It was the {other} who {verb_past} the {target}.",
}

# Draft relative-clause 2x2, pending team validation.
_RELATIVE_CLAUSE: dict[_Cell, str] = {
    (Role.AGENT, Position.FIRST): "The {target} who {verb_past} the {other} smiled.",
    (Role.AGENT, Position.SECOND): "The {other} whom the {target} {verb_past} smiled.",
    (Role.PATIENT, Position.FIRST): "The {target} whom the {other} {verb_past} smiled.",
    (Role.PATIENT, Position.SECOND): "The {other} who {verb_past} the {target} smiled.",
}

# Draft dative 2x2, pending team validation. "Role" here is
# giver/recipient, with the recipient mapped onto the PATIENT slot. Actives use
# the prepositional frame, passives the passivized double-object frame, and the
# theme is fixed ("a letter") so the four cells stay lexically matched.
_DATIVE: dict[_Cell, str] = {
    (Role.AGENT, Position.FIRST): "The {target} {verb_past} a letter to the {other}.",
    (Role.AGENT, Position.SECOND): "The {other} was {verb_past} a letter by the {target}.",
    (Role.PATIENT, Position.FIRST): "The {target} was {verb_past} a letter by the {other}.",
    (Role.PATIENT, Position.SECOND): "The {other} {verb_past} a letter to the {target}.",
}

# All four constructions form clean 2x2s, so no NotImplementedError cells.
_TEMPLATES: dict[Construction, dict[_Cell, str]] = {
    Construction.ACTIVE_PASSIVE: _ACTIVE_PASSIVE,
    Construction.DATIVE: _DATIVE,
    Construction.CLEFT: _CLEFT,
    Construction.RELATIVE_CLAUSE: _RELATIVE_CLAUSE,
}


def _concept_probe(profession: str) -> str:
    """CONCEPT probe naming one profession's cue. Missing cue is fatal rather
    than silently skipped: a family whose participant has no cue cannot be
    scored on the recall control, and dropping it would quietly unbalance the
    counterbalancing."""
    cue = PROFESSION_CUE.get(profession)
    if cue is None:
        raise KeyError(
            f"no PROFESSION_CUE for {profession!r}; every entity used as a "
            "participant needs one (stimuli.vocab.PROFESSION_CUE)"
        )
    return _CONCEPT_PROBE_TEMPLATE.format(cue=cue)


def build_family(
    pair: ConceptPair,
    construction: Construction,
    other_entity: str,
    verb_lemma: str,
    family_index: int,
) -> ItemFamily:
    """Instantiate the matched quadruple for one lexical content.

    ACTIVE_PASSIVE reproduces the proposal's worked example verbatim; the
    other constructions are drafts (see the notes on the template tables).
    Both probes are built once per family, so they are byte-identical across
    cells by construction.
    """
    cells = {
        cell_key(role, position): Stimulus(
            role=role,
            position=position,
            sentence=template.format(target=pair.entity, other=other_entity, verb_past=verb_lemma),
        )
        for (role, position), template in _TEMPLATES[construction].items()
    }
    family = ItemFamily(
        family_id=f"{construction.value}|{pair.pair_id}|{family_index:03d}",
        concept_pair=pair,
        construction=construction,
        other_entity=other_entity,
        verb_lemma=verb_lemma,
        cells=cells,
        role_probe=_ROLE_PROBE_TEMPLATES[construction].format(verb_past=verb_lemma),
        neutral_probe=_NEUTRAL_PROBE,
        recipient_probe=_RECIPIENT_PROBE_TEMPLATES.get(construction, "").format(
            verb_past=verb_lemma
        ),
        concept_probe_entity=_concept_probe(pair.entity),
        concept_probe_other=_concept_probe(other_entity),
        answer_set=AnswerSet(entity=pair.entity, counterpart=pair.counterpart, other=other_entity),
    )
    _validate_family(family)
    return family


def _validate_family(family: ItemFamily) -> None:
    """Assert the design invariant a template typo would silently break.

    Probes are byte-identical across cells trivially (one string per family),
    so the load-bearing check is surface order: the target entity must precede
    the other entity exactly when Position is FIRST.
    """
    target = family.concept_pair.entity
    other = family.other_entity
    for stimulus in family.cells.values():
        sentence = stimulus.sentence.lower()
        # "the <entity>" is unambiguous: every template determinizes both nouns
        # and no vocab entity is a prefix of another (see stimuli.vocab).
        target_first = sentence.index(f"the {target}") < sentence.index(f"the {other}")
        assert target_first == (stimulus.position is Position.FIRST), (
            f"{family.family_id} {cell_key(stimulus.role, stimulus.position)}: "
            f"surface order contradicts Position in {stimulus.sentence!r}"
        )
