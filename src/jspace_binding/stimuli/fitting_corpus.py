"""Fitting corpus: agent/patient exemplars used to fit role directions.

Disjoint from the primary stimulus set at the template level — every frame is a
distinct surface template and the verb pool shares no lemma with
templates.VERBS_BY_CONSTRUCTION, both asserted by tests. Balanced by design:
for each entity and role, half the frames place the entity first and half
second, so a fitted direction cannot be a linear-position direction in
disguise. Each frame exists in an agent/patient mirrored pair, so the two role
classes are lexically matched and the difference-of-means isolates role.

The corpus is shared across injection sites; only the activations differ.
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

    The probes mirror ItemFamily's so a held-out corpus can be scored on the
    same readouts as the primary set. All are optional and load with "" —
    calibration only uses role_probe. recipient_probe is DATIVE-only; the two
    concept probes are RQ2's recall control and are both needed, since scoring
    either alone reintroduces the confounds counterbalancing cancels.
    """

    entity: str
    role: Role
    position: Position
    frame_id: str
    verb: str
    other: str
    sentence: str
    role_probe: str
    recipient_probe: str = ""
    neutral_probe: str = ""
    concept_probe_entity: str = ""
    concept_probe_other: str = ""


def distractor_pool(
    entity: str,
    fitted_entities: tuple[str, ...] = (),
    counterparts: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Professions usable as the distractor opposite `entity`.

    A distractor must never be an entity whose own direction is being fitted.
    If two fitted entities are each other's distractor their sentence sets
    coincide, every A-agent sentence being literally a B-patient sentence, so
    r_B = -r_A exactly and the pairwise cosine is pinned at -1 for any model —
    destroying the filler-general vs entity-specific comparison. Counterparts
    are excluded for the same reason.

    Defaults reproduce the historical pool (everything but the entity).
    """
    banned = {entity, *fitted_entities, *counterparts}
    pool = tuple(e for e in PROFESSION_ENTITIES if e not in banned)
    if not pool:
        raise ValueError(
            f"no distractor left for {entity!r}: fitted={sorted(fitted_entities)} "
            f"counterparts={sorted(counterparts)} exhaust the profession vocabulary; "
            "add professions to stimuli.vocab.PROFESSION_ENTITIES or fit fewer entities"
        )
    return pool


def generate_fitting_corpus(
    entities: tuple[str, ...],
    exemplars_per_role: int,
    counterparts: tuple[str, ...] = (),
) -> list[FittingExample]:
    """Deterministic corpus: for each entity x role, exemplars_per_role
    sentences cycling through frames, verbs, then distractors. Position stays
    balanced whenever exemplars_per_role is a multiple of the 6 per-role frames.

    `entities` doubles as the fitted set, so no entity is another's distractor;
    pass `counterparts` to keep the swap partners out of sentences too. See
    :func:`distractor_pool`.
    """
    if exemplars_per_role % 6 != 0:
        raise ValueError(
            f"exemplars_per_role={exemplars_per_role} must be a multiple of 6 "
            "(the per-role frame count) to keep the corpus position-balanced"
        )
    examples: list[FittingExample] = []
    for entity in entities:
        others = distractor_pool(entity, entities, counterparts)
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
        # Omitted when empty so the fitting corpus (ROLE probe only) keeps its
        # existing on-disk shape and older files stay byte-identical.
        **{
            key: value
            for key, value in (
                ("recipient_probe", ex.recipient_probe),
                ("neutral_probe", ex.neutral_probe),
                ("concept_probe_entity", ex.concept_probe_entity),
                ("concept_probe_other", ex.concept_probe_other),
            )
            if value
        },
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
        # Absent in corpora written before these fields existed, and in the
        # fitting corpus, which carries the ROLE probe only.
        recipient_probe=record.get("recipient_probe", ""),
        neutral_probe=record.get("neutral_probe", ""),
        concept_probe_entity=record.get("concept_probe_entity", ""),
        concept_probe_other=record.get("concept_probe_other", ""),
    )
