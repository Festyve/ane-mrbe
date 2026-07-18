"""Structure contracts for the checked-in hand-written dataset + QC helpers.

The known disjointness problems in the first drop (12 primary-set collisions
via praised/criticized/interviewed; recognized x lawyer leaking between
fitting and eval) are DATA issues for the team to fix, so they are surfaced
by scripts/check_corpus.py and the fit_directions warning rather than
asserted here — these tests pin structure only, plus the QC helpers'
behavior on synthetic cases.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from jspace_binding.config import Config, StimuliConfig
from jspace_binding.stimuli.fitting_corpus import load_fitting_corpus
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.stimuli.qc import (
    find_cross_leaks,
    find_primary_collisions,
    structure_report,
)
from jspace_binding.types import Role

DATA = Path(__file__).resolve().parent.parent / "data" / "handwritten"


def test_fitting_doctor_structure() -> None:
    corpus = load_fitting_corpus(DATA / "fitting_doctor.jsonl")
    report = structure_report(corpus)
    assert report["n"] == 80
    assert report["entities"] == ["doctor"]
    assert report["per_role"][Role.AGENT.value] == {"n": 40, "first": 20, "second": 20}
    assert report["per_role"][Role.PATIENT.value] == {"n": 40, "first": 20, "second": 20}
    assert report["position_balanced"] is True
    assert report["duplicate_sentences"] == []
    assert report["unknown_entities"] == []
    for ex in corpus:
        assert ex.sentence.count("doctor") == 1
        assert ex.other in ex.sentence
        assert ex.verb in ex.role_probe


def test_eval_doctor_structure() -> None:
    corpus = load_fitting_corpus(DATA / "eval_doctor.jsonl")
    report = structure_report(corpus)
    assert report["n"] == 30
    assert report["entities"] == ["doctor"]
    assert report["duplicate_sentences"] == []
    assert report["unknown_entities"] == []
    # 8/7/8/7 by design: the eval grid is not fully position-balanced.
    counts = Counter((ex.role, ex.position) for ex in corpus)
    assert sorted(counts.values()) == [7, 7, 8, 8]
    # The eval set itself must be clean against the primary stimulus set.
    families = generate_families(Config())
    assert find_primary_collisions(corpus, families) == {}


def test_qc_helpers_detect_synthetic_collision_and_leak() -> None:
    families = generate_families(Config(stimuli=StimuliConfig(items_per_cell=1)))
    corpus = load_fitting_corpus(DATA / "fitting_doctor.jsonl")
    # Splice a genuine primary sentence into a copy of the corpus.
    primary_sentence = families[0].cells["agent:first"].sentence
    spiked = corpus + [
        type(corpus[0])(**{**corpus[0].__dict__, "sentence": primary_sentence})
    ]
    collisions = find_primary_collisions(spiked, families)
    assert primary_sentence in collisions
    # Cross-leak detection: any shared sentence between two files is reported.
    assert find_cross_leaks(corpus, corpus)  # identical files leak everywhere
    assert find_cross_leaks(corpus[:1], corpus[1:]) == []