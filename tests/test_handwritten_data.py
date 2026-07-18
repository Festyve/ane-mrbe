"""Structure + cleanliness contracts for the checked-in hand-written dataset.

The first drop had 12 primary-set collisions (praised/criticized/interviewed
shared with the primary verb pool) and a recognized x lawyer leak between
fitting and eval. Those are fixed (the corpus is regenerated from the CSVs
via scripts/csv_to_jsonl.py), and test_doctor_corpus_is_clean below locks
the fix in so a future CSV edit can't silently reintroduce contamination.
The rest pin structure and the QC helpers' behavior on synthetic cases.
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


def test_doctor_corpus_is_clean() -> None:
    """The load-bearing guarantee: the doctor fitting corpus shares no
    sentence with the primary stimulus set (can't fit and test on the same
    data) and none with the eval set (no fit/eval leakage)."""
    fitting = load_fitting_corpus(DATA / "fitting_doctor.jsonl")
    eval_set = load_fitting_corpus(DATA / "eval_doctor.jsonl")
    families = generate_families(Config())  # full 50-items/cell scale
    assert find_primary_collisions(fitting, families) == {}
    assert find_cross_leaks(fitting, eval_set) == []


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