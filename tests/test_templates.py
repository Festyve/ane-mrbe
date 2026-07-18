"""Template and generation contract tests (docs/ARCHITECTURE.md, Testing)."""

from __future__ import annotations

from pathlib import Path

from jspace_binding.config import Config, StimuliConfig
from jspace_binding.stimuli.generate import generate_families, load_families, save_families
from jspace_binding.stimuli.templates import build_family
from jspace_binding.types import (
    AnswerSet,
    ConceptPair,
    Construction,
    ItemFamily,
    Position,
    Role,
    cell_key,
)

# Pinned by the proposal (doctor/lawyer/treated); must be reproduced verbatim.
WORKED_EXAMPLE = {
    (Role.AGENT, Position.FIRST): "The doctor treated the lawyer.",
    (Role.AGENT, Position.SECOND): "The lawyer was treated by the doctor.",
    (Role.PATIENT, Position.FIRST): "The doctor was treated by the lawyer.",
    (Role.PATIENT, Position.SECOND): "The lawyer treated the doctor.",
}


def _worked_family(pair: ConceptPair | None = None, other: str = "lawyer") -> ItemFamily:
    return build_family(
        pair=pair or ConceptPair("doctor", "nurse"),
        construction=Construction.ACTIVE_PASSIVE,
        other_entity=other,
        verb_lemma="treated",
        family_index=0,
    )


def _tiny_config() -> Config:
    return Config(stimuli=StimuliConfig(items_per_cell=1))


def test_active_passive_matches_worked_example_verbatim() -> None:
    family = _worked_family()
    assert set(family.cells) == {cell_key(role, pos) for role in Role for pos in Position}
    for (role, position), sentence in WORKED_EXAMPLE.items():
        assert family.cell(role, position).sentence == sentence


def test_probes_byte_identical_and_entity_free() -> None:
    family = _worked_family()
    sibling = _worked_family(pair=ConceptPair("teacher", "student"), other="judge")
    # One string per family makes byte-identity across cells structural; the
    # load-bearing check is that probes do not leak which entities are present,
    # so families sharing a verb share probes byte-for-byte.
    assert family.role_probe and family.neutral_probe
    assert family.role_probe == sibling.role_probe
    assert family.neutral_probe == sibling.neutral_probe
    for entity in ("doctor", "nurse", "lawyer", "teacher", "student", "judge"):
        assert entity not in family.role_probe
        assert entity not in family.neutral_probe
    assert "treated" in family.role_probe  # the role probe is verb-anchored


def test_answer_set_is_entity_counterpart_other() -> None:
    family = _worked_family()
    assert family.answer_set == AnswerSet(entity="doctor", counterpart="nurse", other="lawyer")
    assert family.answer_set.tokens == ("doctor", "nurse", "lawyer")


def test_generate_families_full_crossing_and_deterministic() -> None:
    config = _tiny_config()
    families = generate_families(config)
    n_pairs = len(config.stimuli.concept_pairs)
    assert len(families) == n_pairs * len(Construction) * config.stimuli.items_per_cell
    assert len({family.family_id for family in families}) == len(families)
    assert generate_families(config) == families


def test_save_load_roundtrip(tmp_path: Path) -> None:
    families = generate_families(_tiny_config())
    path = tmp_path / "stimuli.jsonl"
    save_families(families, path)
    assert load_families(path) == families
