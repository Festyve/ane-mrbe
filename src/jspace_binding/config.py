"""Typed config loaded from configs/*.yaml.

One flat, explicit dataclass per section. Fields defaulting to None are the
open Methods decisions — code that needs them fails loudly with a pointer
here rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from jspace_binding.types import ConceptPair, EditType, InjectionSite, PushSign


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "dummy"  # "dummy" | "qwen_jlens"
    model_id: str = "Qwen/Qwen3.6-27B"  # verified on HF: 64 layers, d_model 5120
    lens_repo: str = "neuronpedia/jacobian-lens"  # per-layer averaged Jacobians J_l
    # The lens repo holds one lens per model (~57 GB in total); lens_subpath scopes
    # BOTH the download and the file search to this model's directory.
    lens_subpath: str | None = "qwen3.6-27b/jlens/Salesforce-wikitext"
    layer_band: tuple[int, int] | None = None  # open: RAW layer band; [L, L] = single layer
    # (source paper: workspace ~= reindexed L38-92 of 100; single-layer swaps mid-band.
    # For 64 layers that converts to raw 24-59, mid-band ~48. The released lens covers
    # raw layers 0-62, so any band inside that range is loadable.)
    alpha: float | None = None  # open: IDENTITY_SWAP scaling (calibrated; None = pure swap)
    push_coefficient: float | None = None  # open: role-push scaling, calibrated on the
    # fitting corpus, NEVER on the primary stimuli (experiments.calibrate)
    jspace_k: int = 16  # sparsity of the J-space pursuit (paper uses k=16 for concept
    # components; median workspace occupancy ~25)
    ablate_k: int = 10  # atoms zeroed by ABLATE_JSPACE (paper's §3.5.2 uses k=10)
    dtype: str = "bfloat16"
    load_in_4bit: bool = False  # Kaggle 2x T4 path (proposal, Compute)
    device_map: str | None = "auto"  # "auto" = accelerate multi-GPU sharding (GPU day);
    # None = plain single-device load (CPU/Mac — "auto" segfaults there)
    dummy_mode: str = "binding"  # "binding" | "bag" — ground truth for the dummy backend


@dataclass(frozen=True)
class StimuliConfig:
    items_per_cell: int = 50  # placeholder pending power calculation
    concept_pairs: tuple[ConceptPair, ...] = (
        ConceptPair("doctor", "nurse"),
        ConceptPair("teacher", "student"),
        ConceptPair("driver", "passenger"),
    )
    # Candidates for the NULL_NON_PARTICIPANT push (see stimuli.vocab for the
    # deviation note re: the proposal's "tuesday" example).
    non_participant_entities: tuple[str, ...] = ("chef", "farmer", "coach")


@dataclass(frozen=True)
class DirectionsConfig:
    """Role-direction fitting (proposal, Methods / Role directions).

    exemplars_per_role is flagged in the proposal as an open parameter pending
    the bootstrap stability pilot — the default is a starting point, and the
    fit script warns whenever stability lands below stability_threshold.
    """

    exemplars_per_role: int = 24  # per entity per role; multiple of 6 (frame count)
    n_bootstrap: int = 200  # resamples for the stability pilot check
    stability_threshold: float = 0.8  # warn below this mean cosine
    variant: str = "fitted"  # "fitted" | "generic_loo" — which direction ROLE_PUSH uses
    seed: int = 0  # shuffled-label + bootstrap seeding
    # "diff_means": unit(mean(agent) - mean(patient)) over J-space components —
    # the proposal's estimator, a DECODE direction. "lre_gradient": unit mean of
    # per-exemplar readout gradients d(z_entity - z_other)/dh at the site
    # (LRE/LRC-style, Chanin et al. 2023) — a STEERING direction. The two need
    # not coincide; a null push under diff_means may be a bad-estimator
    # artifact, which lre_gradient exists to rule in or out.
    estimator: str = "diff_means"  # "diff_means" | "lre_gradient"


@dataclass(frozen=True)
class ExperimentConfig:
    edit_types: tuple[EditType, ...] = (
        EditType.ROLE_PUSH,
        EditType.NO_EDIT,
        EditType.NULL_NON_PARTICIPANT,
        EditType.RANDOM_DIRECTION,
        EditType.SHUFFLED_LABEL_DIRECTION,
        EditType.IDENTITY_SWAP,
    )
    push_signs: tuple[PushSign, ...] = (PushSign.TOWARD_AGENT, PushSign.TOWARD_PATIENT)
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
    fitting_corpus: Path = Path("data/stimuli/fitting_corpus.jsonl")
    directions: Path = Path("data/directions")
    calibration: Path = Path("data/calibration.json")
    results: Path = Path("data/results")
    figures: Path = Path("figures")


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    stimuli: StimuliConfig = field(default_factory=StimuliConfig)
    directions: DirectionsConfig = field(default_factory=DirectionsConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def direction_entities(self) -> tuple[str, ...]:
        """Every entity needing a fitted role direction: the pushed pair
        entities plus the non-participant pool. The canonical definition —
        fitting and calibration must agree on it."""
        return tuple(
            dict.fromkeys(
                [pair.entity for pair in self.stimuli.concept_pairs]
                + list(self.stimuli.non_participant_entities)
            )
        )

    def counterpart_entities(self) -> tuple[str, ...]:
        """The identity-swap counterparts. They are never pushed and never
        appear in a primary sentence (stimuli.templates), so the fitting
        corpus keeps them out of its sentences too — passing these to
        stimuli.fitting_corpus.generate_fitting_corpus prevents a fitted
        entity and its counterpart from being each other's distractor, which
        would pin their direction cosine at -1 (see distractor_pool)."""
        return tuple(dict.fromkeys(pair.counterpart for pair in self.stimuli.concept_pairs))

    @staticmethod
    def from_yaml(path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        model_raw = dict(raw.get("model", {}))
        if model_raw.get("layer_band") is not None:
            model_raw["layer_band"] = tuple(model_raw["layer_band"])

        stimuli_raw = dict(raw.get("stimuli", {}))
        if "concept_pairs" in stimuli_raw:
            stimuli_raw["concept_pairs"] = tuple(
                ConceptPair(entity, counterpart)
                for entity, counterpart in stimuli_raw["concept_pairs"]
            )
        if "non_participant_entities" in stimuli_raw:
            stimuli_raw["non_participant_entities"] = tuple(
                stimuli_raw["non_participant_entities"]
            )

        experiment_raw = dict(raw.get("experiment", {}))
        if "edit_types" in experiment_raw:
            experiment_raw["edit_types"] = tuple(
                EditType(e) for e in experiment_raw["edit_types"]
            )
        if "push_signs" in experiment_raw:
            experiment_raw["push_signs"] = tuple(
                PushSign(s) for s in experiment_raw["push_signs"]
            )
        if "injection_sites" in experiment_raw:
            experiment_raw["injection_sites"] = tuple(
                InjectionSite(s) for s in experiment_raw["injection_sites"]
            )

        paths_raw = {k: Path(v) for k, v in raw.get("paths", {}).items()}

        return Config(
            model=ModelConfig(**model_raw),
            stimuli=StimuliConfig(**stimuli_raw),
            directions=DirectionsConfig(**raw.get("directions", {})),
            experiment=ExperimentConfig(**experiment_raw),
            analysis=AnalysisConfig(**raw.get("analysis", {})),
            paths=PathsConfig(**paths_raw),
        )
