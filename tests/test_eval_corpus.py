"""Contracts for the four-construction held-out evaluation corpus.

The eval tier only means something if it is disjoint from the tier the
directions were fit on AND from the primary set they are pushed on. Those are
the load-bearing assertions here; the rest pin structure and determinism.

The checked-in JSONL is generated (scripts/generate_eval_corpus.py), so
test_checked_in_files_match_generator is the anti-drift lock: hand-editing the
data without regenerating fails here, the same guarantee csv_to_jsonl.py gives
the hand-written corpora.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from jspace_binding.analysis.direction_sanity import template_signature
from jspace_binding.config import Config
from jspace_binding.stimuli.eval_corpus import (
    EVAL_VERBS_BY_CONSTRUCTION,
    eval_frame_templates,
    generate_eval_corpus,
)
from jspace_binding.stimuli.fitting_corpus import (
    frame_templates,
    generate_fitting_corpus,
    load_fitting_corpus,
)
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.types import Construction, Position

DATA = Path(__file__).resolve().parent.parent / "data" / "handwritten"
CHECKED_IN = ("doctor", "nurse")


def _signatures(examples) -> set[str]:
    return {template_signature(e.sentence, e.entity, e.other, e.verb) for e in examples}


def _primary_signatures() -> set[str]:
    return {
        template_signature(
            cell.sentence, fam.concept_pair.entity, fam.other_entity, fam.verb_lemma
        )
        for fam in generate_families(Config())
        for cell in fam.cells.values()
    }


def test_covers_all_four_constructions() -> None:
    corpus = generate_eval_corpus(("doctor",))
    assert len(corpus) == 128  # 4 constructions x 4 cells x 8 verbs
    assert len({ex.frame_id for ex in corpus}) == len(Construction) == 4


def test_role_position_grid_balanced_per_construction() -> None:
    """position_average raises on a role missing a position, so every
    construction must fill all four cells."""
    corpus = generate_eval_corpus(("doctor",))
    by_frame: dict[str, Counter] = {}
    for ex in corpus:
        by_frame.setdefault(ex.frame_id, Counter())[(ex.role, ex.position)] += 1
    for frame, counts in by_frame.items():
        assert len(counts) == 4, f"{frame} missing cells"
        assert len(set(counts.values())) == 1, f"{frame} unbalanced: {counts}"


def test_position_label_matches_surface_order() -> None:
    """A mislabelled position silently reintroduces the word-order confound
    that position-averaging exists to cancel (cf. templates._validate_family)."""
    for ex in generate_eval_corpus(("doctor", "nurse")):
        sentence = ex.sentence.lower()
        entity_first = sentence.index(f"the {ex.entity}") < sentence.index(f"the {ex.other}")
        assert entity_first == (ex.position is Position.FIRST), ex.sentence


def test_templates_disjoint_from_primary_and_fitting() -> None:
    """The whole point of the tier: fit on one set of frames, validate on
    another, push on a third."""
    corpus = generate_eval_corpus(("doctor",))
    eval_sigs = _signatures(corpus)
    assert eval_sigs & _primary_signatures() == set()
    assert eval_sigs & _signatures(generate_fitting_corpus(("doctor",), 24)) == set()
    # Raw template strings must differ too, not just their filled signatures.
    assert set(eval_frame_templates()) & set(frame_templates()) == set()


def test_verbs_disjoint_from_primary_and_fitting_pools() -> None:
    from jspace_binding.stimuli.fitting_corpus import FITTING_VERBS
    from jspace_binding.stimuli.templates import VERBS_BY_CONSTRUCTION

    eval_verbs = {v for pool in EVAL_VERBS_BY_CONSTRUCTION.values() for v in pool}
    primary_verbs = {v for pool in VERBS_BY_CONSTRUCTION.values() for v in pool}
    assert eval_verbs & primary_verbs == set()
    assert eval_verbs & set(FITTING_VERBS) == set()


def test_dative_probe_is_ditransitive() -> None:
    """"Who brought someone?" is ungrammatical; the dative probe carries the
    fixed theme, mirroring templates._ROLE_PROBE_TEMPLATES."""
    corpus = generate_eval_corpus(("doctor",), constructions=(Construction.DATIVE,))
    for ex in corpus:
        assert ex.role_probe == f"Question: Who {ex.verb} a letter to someone? Answer: The"
    other = generate_eval_corpus(("doctor",), constructions=(Construction.ACTIVE_PASSIVE,))
    for ex in other:
        assert ex.role_probe == f"Question: Who {ex.verb} someone? Answer: The"


def test_carries_all_four_probe_types() -> None:
    """The held-out corpus must ask the SAME questions as the primary set, or
    a sanity result would not transfer. Strings are imported from templates.py
    rather than restated, so this also catches drift if a probe is reworded."""
    from jspace_binding.stimuli.templates import _NEUTRAL_PROBE

    for ex in generate_eval_corpus(("doctor", "nurse")):
        assert ex.neutral_probe == _NEUTRAL_PROBE
        # CONCEPT is counterbalanced: both sides are required, and each names
        # its own participant's cue rather than the other's.
        assert ex.concept_probe_entity and ex.concept_probe_other
        assert ex.concept_probe_entity != ex.concept_probe_other
        # RECIPIENT exists only where a recipient reading does.
        is_dative = ex.frame_id == "eval4_dative_double_object"
        assert bool(ex.recipient_probe) is is_dative, ex.sentence
        for probe in (
            ex.role_probe,
            ex.neutral_probe,
            ex.concept_probe_entity,
            ex.concept_probe_other,
        ):
            assert probe.endswith("Answer: The")


def test_concept_probe_requires_a_cue_for_every_participant() -> None:
    """Fatal rather than silently skipped: a participant with no cue cannot be
    scored on the recall control, and dropping it would unbalance the pair."""
    with pytest.raises(KeyError, match="PROFESSION_CUE"):
        generate_eval_corpus(("doctor",), other_entity="cyclist")


def test_generation_is_deterministic() -> None:
    assert generate_eval_corpus(("doctor",)) == generate_eval_corpus(("doctor",))


def test_rejects_other_entity_collision() -> None:
    with pytest.raises(ValueError, match="must differ"):
        generate_eval_corpus(("lawyer",), other_entity="lawyer")


@pytest.mark.parametrize("entity", CHECKED_IN)
def test_checked_in_files_match_generator(entity: str) -> None:
    """Anti-drift: the committed JSONL must be exactly what the generator
    emits, so a hand edit cannot silently diverge from the rule."""
    on_disk = load_fitting_corpus(DATA / f"eval_{entity}_4construction.jsonl")
    assert on_disk == generate_eval_corpus((entity,))
