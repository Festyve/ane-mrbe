#!/usr/bin/env python3
"""Audit fitting/eval corpus disjointness across the expanded dataset files.

`direction_sanity.py` enforces disjointness at run time for the one fit/eval
pair it is handed; this is the repeatable audit over many files at once,
reporting overlap for every fit x eval pair on each contaminating axis:
  - sentence  : exact sentence reuse (weakest holdout, still a hard fail)
  - frame_id  : shared structural frame IDs         (overlapping_templates)
  - template  : shared normalized surface templates (template_signature) —
                catches an old template relabeled with a fresh frame_id
  - entity    : shared target entities — reported as INFO, NOT a failure:
                a held-out set for entity E must, by design, contain E (you
                fit E's direction, then evaluate it on unseen E sentences).

Exit status: 0 if every pair is clean on the failure axes (sentence, frame_id,
template); 1 if any pair overlaps. A JSON summary goes to stdout; the
human-readable per-pair report goes to stderr.

Usage:
    audit_disjointness.py --fit data/stimuli/fitting_corpus.jsonl \
                          --eval data/handwritten/eval_doctor.jsonl
    audit_disjointness.py --fit data/stimuli/*.jsonl \
                          --eval data/handwritten/eval_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jspace_binding.analysis.direction_sanity import overlapping_templates, template_signature
from jspace_binding.stimuli.fitting_corpus import load_fitting_corpus


def _signatures(examples) -> set[str]:
    return {template_signature(e.sentence, e.entity, e.other, e.verb) for e in examples}


def _audit_pair(fit_path: Path, eval_path: Path) -> dict:
    fit = load_fitting_corpus(fit_path)
    ev = load_fitting_corpus(eval_path)

    frame_overlap = overlapping_templates(
        (e.frame_id for e in fit), (e.frame_id for e in ev)
    )
    template_overlap = tuple(sorted(_signatures(fit) & _signatures(ev)))
    sentence_overlap = tuple(sorted({e.sentence for e in fit} & {e.sentence for e in ev}))
    entity_overlap = tuple(sorted({e.entity for e in fit} & {e.entity for e in ev}))

    contaminated = bool(frame_overlap or template_overlap or sentence_overlap)
    return {
        "fit": str(fit_path),
        "eval": str(eval_path),
        "fit_n": len(fit),
        "eval_n": len(ev),
        "sentence_overlap": list(sentence_overlap),
        "frame_id_overlap": list(frame_overlap),
        "template_overlap": list(template_overlap),
        "entity_overlap": list(entity_overlap),  # informational
        "contaminated": contaminated,
    }


def _report(pair: dict) -> None:
    tag = "CONTAMINATED" if pair["contaminated"] else "clean"
    print(
        f"[{tag}] fit={pair['fit']} (n={pair['fit_n']}) "
        f"eval={pair['eval']} (n={pair['eval_n']})",
        file=sys.stderr,
    )
    if pair["sentence_overlap"]:
        print(f"    sentence overlap : {len(pair['sentence_overlap'])} exact", file=sys.stderr)
    if pair["frame_id_overlap"]:
        print(f"    frame_id overlap : {', '.join(pair['frame_id_overlap'])}", file=sys.stderr)
    if pair["template_overlap"]:
        print(f"    template overlap : {len(pair['template_overlap'])}", file=sys.stderr)
        for sig in pair["template_overlap"]:
            print(f"        {sig}", file=sys.stderr)
    # entity overlap is expected by design; surface it but never fail on it
    entities = ", ".join(pair["entity_overlap"]) or "(none)"
    print(f"    entity overlap   : {entities} [ok by design]", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fitting/eval corpus disjointness.")
    parser.add_argument("--fit", nargs="+", required=True, type=Path, help="fitting JSONL(s)")
    parser.add_argument("--eval", nargs="+", required=True, type=Path, help="eval JSONL(s)")
    args = parser.parse_args()

    for p in (*args.fit, *args.eval):
        if not p.exists():
            sys.exit(f"no such file: {p}")

    pairs = [_audit_pair(f, e) for f in args.fit for e in args.eval]
    for pair in pairs:
        _report(pair)

    bad = [p for p in pairs if p["contaminated"]]
    summary = {
        "pairs_audited": len(pairs),
        "contaminated_pairs": len(bad),
        "results": pairs,
    }
    print(json.dumps(summary, indent=2))

    if bad:
        print(
            f"\n{len(bad)} contaminated pair(s): a held-out result on these is invalid. "
            "Rewrite the eval set with frames/templates disjoint from the fitting corpus.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"\nall {len(pairs)} pair(s) disjoint on sentence/frame_id/template.", file=sys.stderr)


if __name__ == "__main__":
    main()
