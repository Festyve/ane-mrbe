"""Held-out EVALUATION corpus spanning all four construction families.
As a note, this generation script was co-authored by Claude Opus 5 and later human reviewed.
Third tier of the dataset, distinct from the two that already exist:

    primary set      (templates.py)      -> where directions are PUSHED
    fitting corpus   (fitting_corpus.py) -> where directions are FIT
    eval corpus      (this module)       -> where directions are VALIDATED

`scripts/direction_sanity.py` refuses an eval set that reuses the fitting
corpus's sentences, frame IDs, or normalized surface templates, so the frames
below are disjoint from BOTH other tiers on all three axes (locked by tests).

Why generated rather than hand-authored via CSV: the hand-written doctor
corpus (data/stimuli/doctor.csv -> scripts/csv_to_jsonl.py) exists because a
human picked idiosyncratic verb/distractor pairings. This corpus is a
deterministic cross-product — construction x cell x verb — so a spreadsheet
would only be a lossy transcription of a rule. Same reasoning that makes the
primary set generated, with CSV as an export rather than a source.

Coverage matters here: the primary experiment pushes ONE fitted direction
across all four constructions, but the hand-written eval set covers
active/passive only. A direction validated on active/passive alone has never
been tested against the syntax it is actually used on. These frames close
that gap.

Note the DummyModel scores only two of the four constructions correctly: its
role inference is lexical ("agent = the participant after a `by` or `whom`,
else the earlier one"), which cannot parse pseudo-clefts or `that`-relatives.
That is a dummy limitation, not a labelling error — the real backend reads
activations and has no such heuristic.
"""

from __future__ import annotations

from jspace_binding.stimuli.fitting_corpus import FittingExample
from jspace_binding.types import Construction, Position, Role

# Disjoint from the primary pool (templates.VERBS_BY_CONSTRUCTION), the
# fitting pool (fitting_corpus.FITTING_VERBS), and the hand-written corpora.
# Every past form doubles as the participle, so the passive frames need no
# conjugation logic.
EVAL_TRANSITIVE_VERBS: tuple[str, ...] = (
    "rescued",
    "alerted",
    "blamed",
    "coached",
    "defended",
    "dismissed",
    "filmed",
    "instructed",
)

# Ditransitives that stay natural with the fixed theme ("a letter") in both
# the double-object and prepositional-passive frames.
EVAL_DATIVE_VERBS: tuple[str, ...] = (
    "brought",
    "delivered",
    "passed",
    "forwarded",
    "returned",
    "read",
    "faxed",
    "posted",
)

EVAL_VERBS_BY_CONSTRUCTION: dict[Construction, tuple[str, ...]] = {
    Construction.ACTIVE_PASSIVE: EVAL_TRANSITIVE_VERBS,
    Construction.CLEFT: EVAL_TRANSITIVE_VERBS,
    Construction.RELATIVE_CLAUSE: EVAL_TRANSITIVE_VERBS,
    Construction.DATIVE: EVAL_DATIVE_VERBS,
}

# The dative's theme is fixed so the four cells stay lexically matched, and it
# is embedded in the probe: "Who brought someone?" is ungrammatical for a
# ditransitive. Mirrors templates._ROLE_PROBE_TEMPLATES.
_THEME = "a letter"

_ROLE_PROBE_TEMPLATES: dict[Construction, str] = {
    Construction.ACTIVE_PASSIVE: "Question: Who {verb} someone? Answer: The",
    Construction.CLEFT: "Question: Who {verb} someone? Answer: The",
    Construction.RELATIVE_CLAUSE: "Question: Who {verb} someone? Answer: The",
    Construction.DATIVE: f"Question: Who {{verb}} {_THEME} to someone? Answer: The",
}

_A, _P = Role.AGENT, Role.PATIENT
_F, _S = Position.FIRST, Position.SECOND

# (frame_id, role, position, template) per construction. {entity} is the
# entity whose direction is validated; {other} the distractor. `position` is
# the surface order of {entity} relative to {other} — asserted in tests,
# because a mislabelled position silently reintroduces the word-order
# confound that position-averaging exists to cancel.
#
# Each family stays WITHIN its construction while changing the surface frame:
#   active_passive  past perfect        (primary: simple past)
#   cleft           pseudo-cleft        (primary: it-cleft)
#   relative_clause `that` + new matrix (primary: who/whom + "smiled")
#   dative          double-object       (primary: prepositional)
#
# active_passive has the least syntactic room — the construction IS the bare
# transitive clause — so its holdout is the weakest of the four by nature.
_FRAMES: dict[Construction, tuple[tuple[str, Role, Position, str], ...]] = {
    Construction.ACTIVE_PASSIVE: (
        ("eval4_active_perfect", _A, _F, "The {entity} had {verb} the {other}."),
        ("eval4_active_perfect", _A, _S, "The {other} had been {verb} by the {entity}."),
        ("eval4_active_perfect", _P, _F, "The {entity} had been {verb} by the {other}."),
        ("eval4_active_perfect", _P, _S, "The {other} had {verb} the {entity}."),
    ),
    Construction.CLEFT: (
        ("eval4_cleft_pseudo", _A, _F, "The one the {entity} {verb} was the {other}."),
        ("eval4_cleft_pseudo", _A, _S, "The one who {verb} the {other} was the {entity}."),
        ("eval4_cleft_pseudo", _P, _F, "The one who {verb} the {entity} was the {other}."),
        ("eval4_cleft_pseudo", _P, _S, "The one the {other} {verb} was the {entity}."),
    ),
    Construction.RELATIVE_CLAUSE: (
        ("eval4_relative_that", _A, _F, "The {entity} that {verb} the {other} left early."),
        ("eval4_relative_that", _A, _S, "The {other} that the {entity} {verb} left early."),
        ("eval4_relative_that", _P, _F, "The {entity} that the {other} {verb} left early."),
        ("eval4_relative_that", _P, _S, "The {other} that {verb} the {entity} left early."),
    ),
    Construction.DATIVE: (
        # Giver -> AGENT, recipient -> PATIENT, matching templates._DATIVE.
        ("eval4_dative_double_object", _A, _F, "The {entity} {verb} the {other} a letter."),
        (
            "eval4_dative_double_object",
            _A,
            _S,
            "A letter was {verb} to the {other} by the {entity}.",
        ),
        (
            "eval4_dative_double_object",
            _P,
            _F,
            "A letter was {verb} to the {entity} by the {other}.",
        ),
        ("eval4_dative_double_object", _P, _S, "The {other} {verb} the {entity} a letter."),
    ),
}


def generate_eval_corpus(
    entities: tuple[str, ...],
    other_entity: str = "lawyer",
    constructions: tuple[Construction, ...] | None = None,
    verbs_per_cell: int | None = None,
) -> list[FittingExample]:
    """Deterministic held-out corpus: entity x construction x cell x verb.

    Every construction contributes all four (role x position) cells for every
    verb, so the role x position grid is balanced by construction — no
    `verbs_per_cell` value can unbalance it. No randomness and no dependence
    on any experiment seed: regeneration is byte-identical, matching
    stimuli.generate's determinism contract.

    `verbs_per_cell` defaults to the full pool (8), giving 128 examples per
    entity across the four constructions.
    """
    constructions = constructions or tuple(Construction)
    examples: list[FittingExample] = []
    for entity in entities:
        if entity == other_entity:
            raise ValueError(
                f"other_entity {other_entity!r} must differ from the fitted entity"
            )
        for construction in constructions:
            verbs = EVAL_VERBS_BY_CONSTRUCTION[construction]
            if verbs_per_cell is not None:
                if not 1 <= verbs_per_cell <= len(verbs):
                    raise ValueError(
                        f"verbs_per_cell={verbs_per_cell} outside 1..{len(verbs)} "
                        f"for {construction.value}"
                    )
                verbs = verbs[:verbs_per_cell]
            probe = _ROLE_PROBE_TEMPLATES[construction]
            for frame_id, role, position, template in _FRAMES[construction]:
                for verb in verbs:
                    examples.append(
                        FittingExample(
                            entity=entity,
                            role=role,
                            position=position,
                            frame_id=frame_id,
                            verb=verb,
                            other=other_entity,
                            sentence=template.format(
                                entity=entity, other=other_entity, verb=verb
                            ),
                            role_probe=probe.format(verb=verb),
                        )
                    )
    return examples


def eval_frame_templates() -> tuple[str, ...]:
    """The raw frame strings, exposed for the template-disjointness tests."""
    return tuple(
        template
        for frames in _FRAMES.values()
        for _, _, _, template in frames
    )
