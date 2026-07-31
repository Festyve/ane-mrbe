"""Contract tests for the role-direction fitting corpus (proposal, Datasets §1)."""

from __future__ import annotations

from pathlib import Path

from jspace_binding.stimuli.fitting_corpus import (
    FITTING_VERBS,
    frame_templates,
    generate_fitting_corpus,
    load_fitting_corpus,
    save_fitting_corpus,
)
from jspace_binding.stimuli.templates import _TEMPLATES as PRIMARY_TEMPLATES  # noqa: PLC2701
from jspace_binding.stimuli.templates import VERBS_BY_CONSTRUCTION
from jspace_binding.types import Position, Role

ENTITIES = ("doctor", "nurse", "chef")


def test_verb_pool_disjoint_from_primary() -> None:
    primary_verbs = {verb for verbs in VERBS_BY_CONSTRUCTION.values() for verb in verbs}
    assert not set(FITTING_VERBS) & primary_verbs


def test_templates_disjoint_from_primary() -> None:
    """Template-level disjointness: no fitting frame equals any primary
    template once both are normalized to the same placeholder names."""
    primary = {
        template.replace("{target}", "{entity}").replace("{verb_past}", "{verb}")
        for cells in PRIMARY_TEMPLATES.values()
        for template in cells.values()
    }
    assert not set(frame_templates()) & primary


def test_no_fitted_entity_is_another_entitys_distractor() -> None:
    """If A is B's distractor and vice versa, every A-agent sentence IS a
    B-patient sentence. The final-token activation depends only on the
    sentence string, so r_B = -r_A exactly and the pairwise cosine is pinned
    at -1 for any model — destroying the filler-general vs entity-specific
    comparison. Observed for real on doctor/nurse before distractor_pool
    excluded fitted entities.
    """
    corpus = generate_fitting_corpus(ENTITIES, exemplars_per_role=12)
    distractors = {ex.other for ex in corpus}
    assert not distractors & set(ENTITIES), (
        f"fitted entities used as distractors: {sorted(distractors & set(ENTITIES))}"
    )

    sentences: dict[str, dict[Role, set[str]]] = {}
    for ex in corpus:
        sentences.setdefault(ex.entity, {Role.AGENT: set(), Role.PATIENT: set()})
        sentences[ex.entity][ex.role].add(ex.sentence)
    for a in ENTITIES:
        for b in ENTITIES:
            if a < b:
                assert sentences[a][Role.AGENT] != sentences[b][Role.PATIENT], (
                    f"{a}/{b} sentence sets mirror each other; their fitted "
                    "directions would be exact negatives"
                )


def test_counterparts_never_appear_in_a_fitting_sentence() -> None:
    """templates.py states "pair.counterpart never appears in any sentence"
    for the primary set; the fitting corpus must honour the same invariant."""
    corpus = generate_fitting_corpus(("doctor",), exemplars_per_role=6, counterparts=("nurse",))
    for ex in corpus:
        assert "nurse" not in ex.sentence, ex.sentence


def test_corpus_is_role_and_position_balanced() -> None:
    corpus = generate_fitting_corpus(ENTITIES, exemplars_per_role=12)
    for entity in ENTITIES:
        for role in Role:
            rows = [ex for ex in corpus if ex.entity == entity and ex.role is role]
            assert len(rows) == 12
            first = sum(1 for ex in rows if ex.position is Position.FIRST)
            assert first == 6  # half FIRST, half SECOND per role


def test_entity_occurs_exactly_once_per_sentence() -> None:
    for ex in generate_fitting_corpus(ENTITIES, exemplars_per_role=6):
        assert ex.sentence.count(ex.entity) == 1
        assert ex.other != ex.entity
        assert ex.other in ex.sentence
        assert ex.verb in ex.role_probe  # calibration probe is verb-anchored


def test_exemplars_must_be_multiple_of_frames() -> None:
    import pytest

    with pytest.raises(ValueError, match="multiple of 6"):
        generate_fitting_corpus(ENTITIES, exemplars_per_role=10)


def test_generation_deterministic_and_roundtrips(tmp_path: Path) -> None:
    corpus = generate_fitting_corpus(ENTITIES, exemplars_per_role=6)
    assert generate_fitting_corpus(ENTITIES, exemplars_per_role=6) == corpus
    path = tmp_path / "fitting.jsonl"
    save_fitting_corpus(corpus, path)
    assert load_fitting_corpus(path) == corpus
