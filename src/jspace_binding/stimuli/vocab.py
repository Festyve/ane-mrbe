"""Entity vocabulary for stimulus generation.

All entities are common single-word professions chosen to be plausibly single
tokens under the Qwen tokenizer. validate_single_token confirms this once the
real tokenizer is available — the analysis reads next-token probabilities, so
a multi-token entity would silently break the answer readout.

No entity here is a prefix of another: templates._validate_family relies on
plain substring search for its surface-order checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

PROFESSION_ENTITIES: tuple[str, ...] = (
    "doctor",
    "nurse",
    "teacher",
    "student",
    "driver",
    "passenger",
    "lawyer",
    "judge",
    "pilot",
    "chef",
    "farmer",
    "coach",
)

# One discriminating cue per profession, for the CONCEPT probe (the recall
# control). The probe asks "Which one {cue}?" of a sentence containing BOTH
# participants, so lexical presence cannot answer it — the model has to know
# what the profession is. See ProbeKind.CONCEPT.
#
# Two constraints, both load-bearing:
#   1. NO SHARED STEM with the profession it identifies ("nursing" would let
#      the model match "nurse" on surface form and skip the semantics).
#   2. Must discriminate WITHIN any (entity, other_entity) pair the generator
#      can produce. Counterparts are excluded from other_entity, so the
#      same-domain collisions (doctor/nurse, teacher/student, driver/passenger)
#      never co-occur as the two participants.
PROFESSION_CUE: dict[str, str] = {
    "doctor": "works in medicine",
    "nurse": "assists on a hospital ward",
    "teacher": "leads a classroom",
    "student": "attends classes to learn",
    "driver": "operates a car",
    "passenger": "rides along without steering",
    "lawyer": "argues cases in court",
    "judge": "presides over a trial",
    "pilot": "flies an aircraft",
    "chef": "cooks in a kitchen",
    "farmer": "grows crops",
    "coach": "trains athletes",
}

# Default non-target participant per concept pair. doctor->nurse uses "lawyer"
# to match the proposal's worked example.
OTHER_ENTITY_BY_PAIR: dict[str, str] = {
    "doctor->nurse": "lawyer",
    "teacher->student": "judge",
    "driver->passenger": "pilot",
}

# Candidates for the NULL_NON_PARTICIPANT control: the pushed direction must
# belong to an entity absent from the sentence, so plan_edit picks the first
# candidate that is not the family's entity, counterpart, or other participant.
# NOTE (deviation from the proposal's "tuesday" example, flagged for team
# review): a day-of-the-week token cannot bear a thematic role, so no r_tuesday
# can be fitted — the role-direction analogue of "absent concept" is an absent
# PROFESSION with its own fitted role direction.
NON_PARTICIPANT_CANDIDATES: tuple[str, ...] = ("chef", "farmer", "coach")


class Tokenizer(Protocol):
    """Duck-typed slice of a HuggingFace tokenizer, so this module never
    imports transformers."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


def validate_single_token(words: Sequence[str], tokenizer: Tokenizer) -> dict[str, bool]:
    """Check each word encodes to exactly one token.

    Words are encoded with a leading space because answer probabilities are
    read as the continuation of a probe ending in "... Answer: The".
    """
    return {
        word: len(tokenizer.encode(f" {word}", add_special_tokens=False)) == 1 for word in words
    }
