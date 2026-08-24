#!/usr/bin/env python3
"""Replication gate: is the J-lens reading this model correctly?

Run before trusting any experiment result. It separates two things RQ1/RQ2/E4
cannot tell apart from their own output — both produce near-chance numbers, and
they call for opposite responses:

    when an experiment reports "role is not in J-space", is that a finding
    about the model, or is the lens simply not reading?

Two machine-checked properties plus an inspection pass:

  1. NON-DEGENERATE — the sparse pursuit selects atoms at all, and the J-space
     component is a non-trivial fraction of the residual. A mis-scaled,
     transposed, or wrong-model lens typically selects zero atoms or
     reconstructs h wholesale.
  2. DIRECTED MODULATION — the "Think about X. Do Y" protocol. X's own token
     must score higher in J-space when the prompt names X than when it names
     something else. Load-bearing because it is a CONTRAST, so a lens that
     merely surfaces frequent tokens cannot pass it.

The selected atoms are PRINTED, not asserted on: what the workspace contains is
this project's research question, and a plumbing gate must not prejudge it.

Uses the backend's internals deliberately — the point is to inspect the lens
machinery the experiments depend on, not a public summary of it.

    python scripts/replication_gate.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jspace_binding.config import Config
from jspace_binding.model.factory import build_model, preflight_or_exit

# Prompts for the inspection pass. Deliberately no expected-token list: what
# these layers actually surface is META-LINGUISTIC ("this sentence", "who is",
# "verb", "reverse"), not the sentence's content words, so an earlier
# content-word assertion failed a lens whose modulation contrast passed cleanly
# on the same run. Content matching tests a hypothesis about workspace CONTENT,
# which is the research question, not a plumbing check.
_INSPECTION_PROBES: tuple[str, ...] = (
    "The doctor treated the lawyer.",
    "The chef prepared the meal in the kitchen.",
    "The pilot landed the aircraft safely.",
)

# Gurnee et al.'s paired-question protocol: same trailing task, different
# concept held in mind. Each entry is (concept, prompt).
_MODULATION_PROMPTS: tuple[tuple[str, str], ...] = (
    ("spider", "Think about a spider. Now count slowly to three."),
    ("piano", "Think about a piano. Now count slowly to three."),
    ("volcano", "Think about a volcano. Now count slowly to three."),
)


def _fraction(component: object, h: object) -> float:
    """||component|| / ||h||: how much of the residual the lens claims."""
    import torch

    return float(torch.linalg.norm(component.float()) / torch.linalg.norm(h.float()))  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--top-k", type=int, default=12, help="atoms to display per prompt")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config)
    preflight_or_exit(model)  # lens reads need no fitted directions
    layer = model._read_layer()  # noqa: SLF001 - inspecting the machinery is the point
    tokenizer = model._tokenizer  # noqa: SLF001

    print(f"model={config.model.model_id}  lens={config.model.lens_repo}")
    print(f"read layer={layer}  jspace_k={config.model.jspace_k}\n")

    report: dict[str, object] = {"layer": layer, "checks": {}}

    # --- 1 + 2: non-degenerate, and semantically recognisable ----------------
    print("=" * 72)
    print("1. NON-DEGENERATE  (+ atoms printed for human inspection)")
    print("=" * 72)
    fractions, observed = [], {}
    for sentence in _INSPECTION_PROBES:
        anchor = len(tokenizer(sentence).input_ids) - 1
        h = model._hidden_at(sentence, anchor)  # noqa: SLF001
        component, atoms = model._jspace_component(h, layer)  # noqa: SLF001
        decoded = [tokenizer.decode([t]).strip() for t in atoms[: args.top_k]]
        fraction = _fraction(component, h)
        fractions.append(fraction)
        observed[sentence] = decoded
        print(f"\n  {sentence!r}")
        print(f"    atoms selected : {len(atoms)}/{config.model.jspace_k}")
        print(f"    ||J|| / ||h||  : {fraction:.3f}")
        print(f"    top atoms      : {decoded}")
    print("\n  NOT auto-checked: what the workspace HOLDS is the research")
    print("  question, so the gate does not prejudge it. Read the atoms.")

    # A working lens claims a real but partial slice of the residual. ~0 means
    # the pursuit found nothing; ~1 means it is reconstructing h wholesale and
    # the "component" is not selective.
    non_degenerate = all(0.01 < f < 0.99 for f in fractions)

    # --- 3: directed modulation ---------------------------------------------
    print("\n" + "=" * 72)
    print("3. DIRECTED MODULATION  (the contrast check)")
    print("=" * 72)
    print("  J-lens score of each concept's own token, per prompt.")
    print("  Want the DIAGONAL to dominate: each concept scores highest when named.\n")

    concepts = [c for c, _ in _MODULATION_PROMPTS]
    token_ids = {}
    for concept in concepts:
        encoded = tokenizer.encode(" " + concept, add_special_tokens=False)
        token_ids[concept] = encoded[0]  # first token is enough for a score

    matrix: dict[str, dict[str, float]] = {}
    for concept, prompt in _MODULATION_PROMPTS:
        anchor = len(tokenizer(prompt).input_ids) - 1
        h = model._hidden_at(prompt, anchor)  # noqa: SLF001
        scores = model._lens_scores(h.to(model._lens_device()), layer)  # noqa: SLF001
        matrix[concept] = {c: float(scores[token_ids[c]]) for c in concepts}

    header = "  prompt \\ token   " + "".join(f"{c:>12s}" for c in concepts)
    print(header)
    print("  " + "-" * (len(header) - 2))
    diagonal_wins = 0
    for concept in concepts:
        row = matrix[concept]
        best = max(row, key=row.get)  # type: ignore[arg-type]
        diagonal_wins += best == concept
        cells = "".join(f"{row[c]:>12.2f}" for c in concepts)
        print(f"  {concept:16s}{cells}   {'<- OK' if best == concept else '<- MISS'}")

    modulation = diagonal_wins >= 2  # majority of prompts surface their own concept

    # --- verdict -------------------------------------------------------------
    report["checks"] = {
        "non_degenerate": non_degenerate,
        "directed_modulation": modulation,
        "jspace_fraction_of_residual": fractions,
        "diagonal_wins": diagonal_wins,
        "observed_atoms": observed,
    }
    passed = non_degenerate and modulation
    report["verdict"] = "PASS" if passed else "FAIL"

    print("\n" + "=" * 72)
    print(f"  non-degenerate      : {'PASS' if non_degenerate else 'FAIL'}")
    print(f"  directed modulation : {'PASS' if modulation else 'FAIL'}")
    print("=" * 72)
    if passed:
        print("GATE PASS — the lens reads this model. A near-chance experiment")
        print("result is therefore about the MODEL, not about the plumbing.")
    else:
        print("GATE FAIL — the lens is not reading correctly at this layer.")
        print("Experiment nulls are UNINTERPRETABLE until this passes. Check the")
        print("layer band first (config model.layer_band), then the lens artifact.")

    out = Path(config.paths.results) / "replication_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
