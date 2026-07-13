"""Typed config loaded from configs/*.yaml.

One flat, explicit dataclass per section. Fields defaulting to None are the
open Methods decisions — code that needs them fails loudly with a pointer
here rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from jspace_binding.types import ConceptPair, EditType, InjectionSite


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "dummy"  # "dummy" | "qwen_jlens"
    model_id: str = "Qwen/Qwen3.6-27B"  # TODO: verify exact HF id against Neuronpedia lens
    lens_repo: str = "anthropics/jacobian-lens"
    layer_band: tuple[int, int] | None = None  # open: workspace band for Qwen3.6-27B
    alpha: float | None = None  # open: swap scaling; None = coordinate-swap default
    dtype: str = "bfloat16"
    dummy_mode: str = "binding"  # "binding" | "bag" — ground truth for the dummy backend


@dataclass(frozen=True)
class StimuliConfig:
    items_per_cell: int = 50  # placeholder pending power calculation (Elizabeth/Reya)
    concept_pairs: tuple[ConceptPair, ...] = (
        ConceptPair("doctor", "nurse"),
        ConceptPair("teacher", "student"),
        ConceptPair("driver", "passenger"),
    )
    non_participant_concept: str = "tuesday"


@dataclass(frozen=True)
class ExperimentConfig:
    edit_types: tuple[EditType, ...] = (
        EditType.REAL,
        EditType.NO_EDIT,
        EditType.NULL_NON_PARTICIPANT,
        EditType.RANDOM_DIRECTION,
    )
    injection_sites: tuple[InjectionSite, ...] = (
        InjectionSite.FINAL_TOKEN,
        InjectionSite.ENTITY_TOKEN,
    )
    seed: int = 0


@dataclass(frozen=True)
class AnalysisConfig:
    n_bootstrap: int = 10_000
    n_permutation: int = 10_000
    alpha_level: float = 0.05
    ci_level: float = 0.95


@dataclass(frozen=True)
class PathsConfig:
    stimuli: Path = Path("data/stimuli/stimuli.jsonl")
    results: Path = Path("data/results")
    figures: Path = Path("figures")


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    stimuli: StimuliConfig = field(default_factory=StimuliConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        model_raw = dict(raw.get("model", {}))
        if model_raw.get("layer_band") is not None:
            model_raw["layer_band"] = tuple(model_raw["layer_band"])

        stimuli_raw = dict(raw.get("stimuli", {}))
        if "concept_pairs" in stimuli_raw:
            stimuli_raw["concept_pairs"] = tuple(
                ConceptPair(source, target) for source, target in stimuli_raw["concept_pairs"]
            )

        experiment_raw = dict(raw.get("experiment", {}))
        if "edit_types" in experiment_raw:
            experiment_raw["edit_types"] = tuple(
                EditType(e) for e in experiment_raw["edit_types"]
            )
        if "injection_sites" in experiment_raw:
            experiment_raw["injection_sites"] = tuple(
                InjectionSite(s) for s in experiment_raw["injection_sites"]
            )

        paths_raw = {k: Path(v) for k, v in raw.get("paths", {}).items()}

        return Config(
            model=ModelConfig(**model_raw),
            stimuli=StimuliConfig(**stimuli_raw),
            experiment=ExperimentConfig(**experiment_raw),
            analysis=AnalysisConfig(**raw.get("analysis", {})),
            paths=PathsConfig(**paths_raw),
        )
