"""Corpus QC: the disjointness checks the design depends on.

The fitting corpus must be disjoint from the primary stimulus set — fitting and
testing on the same sentences contaminates the causal test — and an evaluation
set must not leak fitting sentences. scripts/check_corpus.py is the CLI, and
fit_directions warns when handed a colliding corpus.
"""

from __future__ import annotations

from collections import Counter

from jspace_binding.stimuli.fitting_corpus import FittingExample
from jspace_binding.stimuli.vocab import PROFESSION_ENTITIES
from jspace_binding.types import ItemFamily, Position, Role


def primary_sentences(families: list[ItemFamily]) -> dict[str, str]:
    """sentence -> family_id over every cell of the primary stimulus set."""
    out: dict[str, str] = {}
    for family in families:
        for stimulus in family.cells.values():
            out.setdefault(stimulus.sentence, family.family_id)
    return out


def find_primary_collisions(
    corpus: list[FittingExample], families: list[ItemFamily]
) -> dict[str, str]:
    """Exact corpus sentences that also occur in the primary set
    (sentence -> colliding family_id). Must be empty for a valid fit."""
    primary = primary_sentences(families)
    return {ex.sentence: primary[ex.sentence] for ex in corpus if ex.sentence in primary}


def find_cross_leaks(
    corpus_a: list[FittingExample], corpus_b: list[FittingExample]
) -> list[str]:
    """Sentences appearing in both corpora (e.g. fitting vs eval leakage)."""
    sentences_b = {ex.sentence for ex in corpus_b}
    return sorted({ex.sentence for ex in corpus_a if ex.sentence in sentences_b})


def structure_report(corpus: list[FittingExample]) -> dict[str, object]:
    """Counts, balance, and vocabulary hygiene for one corpus file.

    missing_roles lists (entity, role) combinations with zero exemplars — a
    difference-of-means fit needs both roles for every entity, so any entry
    here means the corpus cannot fit that entity's direction at all (which
    position_balanced alone would not catch: a one-role corpus is trivially
    "balanced").
    """
    balance = Counter((ex.role.value, ex.position.value) for ex in corpus)
    per_role = {
        role.value: {
            "n": sum(1 for ex in corpus if ex.role is role),
            "first": balance[(role.value, Position.FIRST.value)],
            "second": balance[(role.value, Position.SECOND.value)],
        }
        for role in Role
    }
    entities = sorted({ex.entity for ex in corpus})
    covered = {(ex.entity, ex.role) for ex in corpus}
    missing_roles = sorted(
        f"{entity}:{role.value}"
        for entity in entities
        for role in Role
        if (entity, role) not in covered
    )
    duplicate_sentences = sorted(
        sentence
        for sentence, count in Counter(ex.sentence for ex in corpus).items()
        if count > 1
    )
    unknown_entities = sorted(
        {e for ex in corpus for e in (ex.entity, ex.other) if e not in PROFESSION_ENTITIES}
    )
    position_balanced = all(
        stats["first"] == stats["second"] for stats in per_role.values() if stats["n"]
    )
    return {
        "n": len(corpus),
        "entities": entities,
        "per_role": per_role,
        "position_balanced": position_balanced,
        "missing_roles": missing_roles,
        "duplicate_sentences": duplicate_sentences,
        "unknown_entities": unknown_entities,
    }
