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
