"""Fitting corpus: agent/patient exemplars used to fit role directions.

Disjoint from the primary stimulus set at the template level (proposal,
Datasets §1): every frame here is a distinct surface template from the four
primary constructions, and the verb pool shares no lemma with
templates.VERBS_BY_CONSTRUCTION. tests/test_fitting_corpus.py asserts both.
(The agent-second / patient-first frames are necessarily passive-like — English
has no other way to put an agent late — but their added material keeps the
template strings distinct; flagged for team review.)

The corpus is balanced by design: for each entity and each role, half the
frames place the entity FIRST and half SECOND, so a fitted difference-of-means
direction cannot be a linear-position direction in disguise.

Each frame exists in an agent/patient mirrored pair (same surface template,
entity and other swapped), so the two role classes are lexically matched and
the difference-of-means isolates role.

r_entity is fitted separately per injection site (final-token activations vs
entity-token activations; proposal, Methods) — the corpus itself is shared,
the activations differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jspace_binding.stimuli.vocab import PROFESSION_ENTITIES
from jspace_binding.types import Position, Role

# Disjoint from every primary verb (templates.VERBS_BY_CONSTRUCTION). All past
# forms double as participles, so frames need no conjugation logic.
FITTING_VERBS: tuple[str, ...] = (
    "helped",
    "thanked",
    "visited",
    "called",
    "warned",
    "greeted",
    "followed",
    "trained",
    "paid",
    "described",
)

# (frame_id, entity_role, entity_position, template). {entity} is the entity
# whose direction is being fitted; {other} the distractor participant. Agent
# and patient frames come in mirrored pairs (same surface template, slots
# swapped), so the two role classes are lexically matched.
_A, _P = Role.AGENT, Role.PATIENT
_F, _S = Position.FIRST, Position.SECOND
_FRAMES: tuple[tuple[str, Role, Position, str], ...] = (
    ("agent_first_a", _A, _F, "The {entity} {verb} the {other} this morning."),
    ("agent_first_b", _A, _F, "Everyone knows the {entity} {verb} the {other}."),
    ("agent_first_c", _A, _F, "Apparently the {entity} {verb} the {other} at work."),
    ("agent_second_a", _A, _S, "After lunch, the {other} was {verb} by the {entity}."),
    ("agent_second_b", _A, _S, "It turned out the {other} had been {verb} by the {entity}."),
    ("agent_second_c", _A, _S, "Reports say the {other} was recently {verb} by the {entity}."),
    ("patient_second_a", _P, _S, "The {other} {verb} the {entity} this morning."),
    ("patient_second_b", _P, _S, "Everyone knows the {other} {verb} the {entity}."),
    ("patient_second_c", _P, _S, "Apparently the {other} {verb} the {entity} at work."),
    ("patient_first_a", _P, _F, "After lunch, the {entity} was {verb} by the {other}."),
    ("patient_first_b", _P, _F, "It turned out the {entity} had been {verb} by the {other}."),
    ("patient_first_c", _P, _F, "Reports say the {entity} was recently {verb} by the {other}."),
)


@dataclass(frozen=True)
class FittingExample:
    """One fitting sentence: the entity in a known role at a known position.

    role_probe mirrors the primary role-probe format so push-coefficient
    calibration (experiments.calibrate) can run on fitting sentences with the
    same readout as the primary experiment — the proposal requires calibrating
    on the fitting corpus, never on the primary stimuli.
    """

    entity: str
    role: Role
    position: Position
    frame_id: str
    verb: str
    other: str
    sentence: str
    role_probe: str


def generate_fitting_corpus(
    entities: tuple[str, ...],
    exemplars_per_role: int,
) -> list[FittingExample]:
    """Deterministic corpus: for each entity x role, exemplars_per_role
    sentences cycling through frames (fastest), verbs, then distractors —
    position stays balanced whenever exemplars_per_role is a multiple of the
    6 per-role frames.

    No randomness and no dependence on any experiment seed: regeneration is
    byte-identical, matching stimuli.generate's determinism contract.
    """
    if exemplars_per_role % 6 != 0:
        raise ValueError(
            f"exemplars_per_role={exemplars_per_role} must be a multiple of 6 "
            "(the per-role frame count) to keep the corpus position-balanced"
        )
    examples: list[FittingExample] = []
    for entity in entities:
        others = tuple(e for e in PROFESSION_ENTITIES if e != entity)
        for role in Role:
            frames = [f for f in _FRAMES if f[1] is role]
            for index in range(exemplars_per_role):
                frame_id, _, position, template = frames[index % len(frames)]
                verb = FITTING_VERBS[(index // len(frames)) % len(FITTING_VERBS)]
                other = others[(index // (len(frames) * len(FITTING_VERBS))) % len(others)]
                examples.append(
                    FittingExample(
                        entity=entity,
                        role=role,
                        position=position,
                        frame_id=frame_id,
                        verb=verb,
                        other=other,
                        sentence=template.format(entity=entity, other=other, verb=verb),
                        role_probe=f"Question: Who {verb} someone? Answer: The",
                    )
                )
    return examples


def frame_templates() -> tuple[str, ...]:
    """The raw frame strings, exposed for the template-disjointness test."""
    return tuple(template for _, _, _, template in _FRAMES)


def save_fitting_corpus(examples: list[FittingExample], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(_to_record(ex)) + "\n")


def load_fitting_corpus(path: str | Path) -> list[FittingExample]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [_from_record(json.loads(line)) for line in f if line.strip()]


def _to_record(ex: FittingExample) -> dict[str, Any]:
    return {
        "entity": ex.entity,
        "role": ex.role.value,
        "position": ex.position.value,
        "frame_id": ex.frame_id,
        "verb": ex.verb,
        "other": ex.other,
        "sentence": ex.sentence,
        "role_probe": ex.role_probe,
    }


def _from_record(record: dict[str, Any]) -> FittingExample:
    return FittingExample(
        entity=record["entity"],
        role=Role(record["role"]),
        position=Position(record["position"]),
        frame_id=record["frame_id"],
        verb=record["verb"],
        other=record["other"],
        sentence=record["sentence"],
        role_probe=record["role_probe"],
    )
