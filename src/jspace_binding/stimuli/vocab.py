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

# One discriminating cue per profession, for the CONCEPT probe. Two constraints:
# no shared stem with the profession it identifies (or the model matches on
# surface form), and it must discriminate within any (entity, other_entity) pair
# the generator can produce.
PROFESSION_CUE: dict[str, str] = {
    # Deliberately weak, and read this before "improving" any cue here. The
    # stronger "diagnoses illness" lifts the margin (+1.42 -> +1.76) but takes
    # role-blindness from 1.54 to 3.95 SEM: an ACTION cue aligned with the
    # sentence verb favours whichever participant is performing it, which is
    # exactly the role information the control must not see. Verify any
    # replacement with scripts/check_concept_probe.py.
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

# Default non-target participant per concept pair. The last three back
# configs/expanded_pairs.yaml (6 leave-one-pair-out folds). Counterpart reuse
# across pairs is safe: a counterpart never appears in any sentence.
OTHER_ENTITY_BY_PAIR: dict[str, str] = {
    "doctor->nurse": "lawyer",
    "teacher->student": "judge",
    "driver->passenger": "pilot",
    "lawyer->judge": "chef",
    "pilot->nurse": "farmer",
    "coach->student": "judge",
}

# Candidates for the NULL_NON_PARTICIPANT control: plan_edit picks the first
# that is not the family's entity, counterpart, or other participant. An absent
# PROFESSION rather than the proposal's "tuesday", because a day-of-the-week
# token cannot bear a thematic role and so has no fittable role direction.
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
