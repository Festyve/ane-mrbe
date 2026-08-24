#!/usr/bin/env python3
"""Execute the ENTIRE real-backend pipeline on a tiny model with a synthetic lens.

Usage: smoke_test.py [--model-id Qwen/Qwen2.5-0.5B] [--out DIR]

The dummy backend validates the ANALYSIS against ground truth; this validates
the BACKEND CODE PATHS — model and lens loading, J-lens vectors, sparse
pursuit, edit hooks, token indexing, direction fitting, calibration, and all
three runners — by running them on a small open model.

What makes it principled rather than fake: the logit lens is exactly the J-lens
with J_l = identity (Gurnee et al. 2026 §2.4), so a fabricated artifact of
identity Jacobians makes the backend run a real, if weak, version of the
experiment. RESULT NUMBERS ARE NOT SCIENCE; PASS means every stage executed and
produced well-formed output.

Needs the heavy extras: pip install '.[model]' (CPU is fine; a 0.5B model
runs the whole thing in minutes).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from jspace_binding.config import (
    AnalysisConfig,
    Config,
    DirectionsConfig,
    ExperimentConfig,
    ModelConfig,
    PathsConfig,
    StimuliConfig,
)
from jspace_binding.directions.fit import fit_all, save_directions
from jspace_binding.experiments.calibrate import (
    calibrate_identity_alpha,
    calibrate_push_coefficient,
)
from jspace_binding.experiments.primary import analyze, run_primary
from jspace_binding.experiments.rq1_probe import run_rq1
from jspace_binding.experiments.rq2_ablation import run_rq2
from jspace_binding.stimuli.fitting_corpus import generate_fitting_corpus
from jspace_binding.stimuli.generate import generate_families
from jspace_binding.stimuli.vocab import (
    OTHER_ENTITY_BY_PAIR,
    PROFESSION_ENTITIES,
    validate_single_token,
)
from jspace_binding.types import ConceptPair, InjectionSite, Role


def _stage(name: str) -> None:
    print(f"\n=== {name} ===", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backend smoke test on a tiny model.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--out", type=Path, default=Path("data/smoke"))
    args = parser.parse_args()

    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError:
        sys.exit("smoke test needs the heavy extras: pip install '.[model]'")

    _stage(f"model config: {args.model_id}")
    hf_config = AutoConfig.from_pretrained(args.model_id)
    # Multimodal checkpoints (e.g. Qwen3.6-27B, a *ForConditionalGeneration)
    # nest the text tower's dims under .text_config and expose neither at the
    # top level, so read through to it when present.
    text_config = getattr(hf_config, "text_config", hf_config)
    d_model = int(text_config.hidden_size)
    n_layers = int(text_config.num_hidden_layers)
    # Mid-band single layer, mirroring the source paper's single-layer swaps
    # (~reindexed L75 of 100).
    layer = max(1, round(0.75 * n_layers) - 1)
    print(f"d_model={d_model} n_layers={n_layers} -> layer_band=[{layer},{layer}]",
          file=sys.stderr)

    _stage("tokenizer vocab check")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    ok = validate_single_token(PROFESSION_ENTITIES, tokenizer)
    bad = sorted(w for w, good in ok.items() if not good)
    if bad:
        print(f"multi-token under this tokenizer (excluded): {bad}", file=sys.stderr)
    candidate_pairs = [
        ConceptPair(entity, counterpart)
        for entity, counterpart in (("doctor", "nurse"), ("teacher", "student"),
                                    ("driver", "passenger"))
        if ok.get(entity)
        and ok.get(counterpart)
        and ok.get(OTHER_ENTITY_BY_PAIR[f"{entity}->{counterpart}"])
    ]
    if len(candidate_pairs) < 2:
        sys.exit(
            f"fewer than 2 usable concept pairs under {args.model_id}'s tokenizer — "
            "try a different --model-id"
        )
    pairs = tuple(candidate_pairs[:2])
    non_participants = tuple(
        e for e in ("chef", "farmer", "coach") if ok.get(e)
    ) or ("chef",)

    _stage("synthetic lens artifact (identity Jacobians == logit lens)")
    lens_dir = args.out / "lens"
    lens_dir.mkdir(parents=True, exist_ok=True)
    identity = np.eye(d_model, dtype=np.float32)
    np.savez(lens_dir / "jacobian_lens.npz", **{f"layer_{layer}": identity})
    print(f"wrote {lens_dir / 'jacobian_lens.npz'}", file=sys.stderr)

    config = Config(
        model=ModelConfig(
            backend="qwen_jlens",
            model_id=args.model_id,
            lens_repo=str(lens_dir),
            lens_subpath=None,  # the synthetic lens is written flat into lens_dir,
            # not in the published {model}/jlens/{corpus}/ layout
            layer_band=(layer, layer),
            jspace_k=8,
            ablate_k=5,
            dtype="float32",  # CPU-safe
            device_map=None,  # plain CPU load — "auto" segfaults on Apple Silicon
        ),
        stimuli=StimuliConfig(
            items_per_cell=2,
            concept_pairs=pairs,
            non_participant_entities=non_participants,
        ),
        directions=DirectionsConfig(exemplars_per_role=6, n_bootstrap=20),
        experiment=ExperimentConfig(injection_sites=(InjectionSite.FINAL_TOKEN,)),
        analysis=AnalysisConfig(n_bootstrap=200, n_permutation=200),
        paths=PathsConfig(
            stimuli=args.out / "stimuli.jsonl",
            fitting_corpus=args.out / "fitting_corpus.jsonl",
            directions=args.out / "directions",
            calibration=args.out / "calibration.json",
            results=args.out / "results",
            figures=args.out / "figures",
        ),
    )

    _stage("backend construction + preflight (loads model + lens)")
    from jspace_binding.model.qwen_jlens import QwenJLensModel

    model = QwenJLensModel(
        config.model,
        directions_dir=config.paths.directions,
        direction_variant=config.directions.variant,
    )
    model._ensure_ready()
    print("model + lens loaded", file=sys.stderr)

    _stage("direction fitting (sparse pursuit -> diff-of-means)")
    entities = config.direction_entities()
    corpus = generate_fitting_corpus(
        entities,
        config.directions.exemplars_per_role,
        counterparts=config.counterpart_entities(),
    )
    activations = {}
    for entity in entities:
        rows = {role: [] for role in Role}
        for ex in corpus:
            if ex.entity == entity:
                rows[ex.role].append(
                    model.fitting_activation(ex.sentence, entity, InjectionSite.FINAL_TOKEN)
                )
        activations[entity] = (
            np.asarray(rows[Role.AGENT], dtype=float),
            np.asarray(rows[Role.PATIENT], dtype=float),
        )
    directions = fit_all(
        activations, InjectionSite.FINAL_TOKEN,
        n_bootstrap=config.directions.n_bootstrap, seed=0,
    )
    save_directions(directions, config.paths.directions)
    for i, entity in enumerate(directions.entities):
        print(f"  r_{entity}: stability={float(directions.stability[i]):.3f}",
              file=sys.stderr)

    _stage("calibration (tiny grids; thresholds disabled — exercising code, not science)")
    push = calibrate_push_coefficient(
        model, corpus, grid=(1.0, 4.0), min_logit_shift=-100.0, max_examples=4
    )
    families = generate_families(config)
    try:
        alpha_value = calibrate_identity_alpha(
            model, families, grid=(1.0,), min_prob_shift=-1.0, max_families=2
        ).value
    except ValueError as exc:
        # A tiny model's swap may genuinely fail to lower P(entity); the code
        # path still executed, which is all the smoke test asserts.
        print(f"  note: alpha search found no passing value ({exc}); using 1.0",
              file=sys.stderr)
        alpha_value = 1.0
    print(f"  push_coefficient={push.value} alpha={alpha_value}", file=sys.stderr)
    config = replace(
        config,
        model=replace(config.model, push_coefficient=push.value, alpha=alpha_value),
    )
    model.config = config.model

    _stage(f"primary sweep ({len(families)} families) + analysis")
    trials = run_primary(config, model, families)
    summary = analyze(config, trials)
    print(f"  {len(trials)} trials | pooled BS={summary['pooled']['mean']:.3f} | "
          f"verdict={summary['verdict']['outcome']}", file=sys.stderr)

    _stage("RQ1 probes")
    rq1 = run_rq1(config, model, families)
    _stage("RQ2 ablations")
    rq2 = run_rq2(config, model, families)

    print("\nSMOKE TEST PASS — every backend stage executed.", file=sys.stderr)
    print("(Numbers are NOT science: tiny model, identity lens, tiny n.)", file=sys.stderr)
    print(json.dumps({
        "model": args.model_id, "layer": layer, "n_trials": len(trials),
        "pooled_bs": summary["pooled"]["mean"],
        "verdict": summary["verdict"]["outcome"],
        "rq1_sites": list(rq1["sites"]),
        "rq2_involved": rq2["workspace_causally_involved"],
    }, indent=2))


if __name__ == "__main__":
    main()
