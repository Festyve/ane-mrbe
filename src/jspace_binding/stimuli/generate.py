"""Deterministic stimulus generation: the full pair x construction x item crossing.

There is no randomness in stimuli: family_index selects the verb (fastest) and
other entity (slower) round-robin from fixed pools, so the verb x other-entity
grid is covered evenly and regeneration is byte-identical. Determinism holds
independently of config.experiment.seed, which stimuli never consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jspace_binding.config import Config
from jspace_binding.stimuli.templates import VERBS_BY_CONSTRUCTION, build_family
from jspace_binding.stimuli.vocab import OTHER_ENTITY_BY_PAIR, PROFESSION_ENTITIES
from jspace_binding.types import (
    AnswerSet,
    ConceptPair,
    Construction,
    ItemFamily,
    Position,
    Role,
    Stimulus,
)


def _other_entity_pool(pair: ConceptPair) -> tuple[str, ...]:
    """Default other-entity first, then the rest of the profession vocab."""
    if pair.pair_id not in OTHER_ENTITY_BY_PAIR:
        raise KeyError(
            f"No default other-entity for pair {pair.pair_id!r}; "
            "add it to stimuli.vocab.OTHER_ENTITY_BY_PAIR"
        )
    default = OTHER_ENTITY_BY_PAIR[pair.pair_id]
    used = {pair.entity, pair.counterpart, default}
    return (default, *(e for e in PROFESSION_ENTITIES if e not in used))


def generate_families(config: Config) -> list[ItemFamily]:
    """Build items_per_cell families for every concept pair x construction."""
    families: list[ItemFamily] = []
    for pair in config.stimuli.concept_pairs:
        pool = _other_entity_pool(pair)
        for construction in Construction:
            verbs = VERBS_BY_CONSTRUCTION[construction]
            for family_index in range(config.stimuli.items_per_cell):
                families.append(
                    build_family(
                        pair=pair,
                        construction=construction,
                        other_entity=pool[(family_index // len(verbs)) % len(pool)],
                        verb_lemma=verbs[family_index % len(verbs)],
                        family_index=family_index,
                    )
                )
    return families


def save_families(families: list[ItemFamily], path: str | Path) -> None:
    """Write one JSON record per line, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for family in families:
            f.write(json.dumps(_to_record(family)) + "\n")


def load_families(path: str | Path) -> list[ItemFamily]:
    """Inverse of save_families."""
    with Path(path).open("r", encoding="utf-8") as f:
        return [_from_record(json.loads(line)) for line in f if line.strip()]


def _to_record(family: ItemFamily) -> dict[str, Any]:
    pair, answers = family.concept_pair, family.answer_set
    return {
        "family_id": family.family_id,
        "concept_pair": {"entity": pair.entity, "counterpart": pair.counterpart},
        "construction": family.construction.value,
        "other_entity": family.other_entity,
        "verb_lemma": family.verb_lemma,
        "cells": {
            key: {"role": s.role.value, "position": s.position.value, "sentence": s.sentence}
            for key, s in family.cells.items()
        },
        "role_probe": family.role_probe,
        "neutral_probe": family.neutral_probe,
        "recipient_probe": family.recipient_probe,
        "answer_set": None
        if answers is None
        else {
            "entity": answers.entity,
            "counterpart": answers.counterpart,
            "other": answers.other,
        },
    }


def _from_record(record: dict[str, Any]) -> ItemFamily:
    answers = record["answer_set"]
    return ItemFamily(
        family_id=record["family_id"],
        concept_pair=ConceptPair(**record["concept_pair"]),
        construction=Construction(record["construction"]),
        other_entity=record["other_entity"],
        verb_lemma=record["verb_lemma"],
        cells={
            key: Stimulus(
                role=Role(cell["role"]),
                position=Position(cell["position"]),
                sentence=cell["sentence"],
            )
            for key, cell in record["cells"].items()
        },
        role_probe=record["role_probe"],
        neutral_probe=record["neutral_probe"],
        # .get: stimuli JSONL written before the recipient probe existed still loads.
        recipient_probe=record.get("recipient_probe", ""),
        answer_set=None if answers is None else AnswerSet(**answers),
    )
