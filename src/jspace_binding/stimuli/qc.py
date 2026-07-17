"""Corpus QC: the disjointness checks the proposal's design depends on.

The role-direction fitting corpus must be disjoint from the primary stimulus
set (proposal, Datasets §1 — fitting and testing on the same sentences would
contaminate the causal test), and any evaluation set must not leak fitting
sentences. These helpers make the checks mechanical; scripts/check_corpus.py
is the CLI, and fit_directions warns when handed a colliding corpus.

Findings on the first hand-written drop (2026-07, doctor corpus): the
active/passive frames necessarily match the primary templates, which is
tolerable at the lexical level — but three of its verbs (praised, criticized,
interviewed) are ALSO in the primary verb pool, producing 12 exact sentence
collisions with generated primary items, and the recognized x lawyer combo
appears in both the fitting corpus and the eval set (4 leaked sentences).
Remedy: keep the handwritten verbs out of templates.VERBS_BY_CONSTRUCTION
(or vice versa), and de-duplicate combos across fitting/eval files.
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
    """Counts, balance, and vocabulary hygiene for one corpus file."""
    balance = Counter((ex.role.value, ex.position.value) for ex in corpus)
    per_role = {
        role.value: {
            "n": sum(1 for ex in corpus if ex.role is role),
            "first": balance[(role.value, Position.FIRST.value)],
            "second": balance[(role.value, Position.SECOND.value)],
        }
        for role in Role
    }
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
        "entities": sorted({ex.entity for ex in corpus}),
        "per_role": per_role,
        "position_balanced": position_balanced,
        "duplicate_sentences": duplicate_sentences,
        "unknown_entities": unknown_entities,
    }
