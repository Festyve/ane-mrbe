"""Validate RQ2's CONCEPT recall control against a real model.

Scores the counterbalanced CONCEPT probe exactly as `experiments.rq2_ablation`
does, alongside the NEUTRAL probe it replaced, answering the three questions a
recall control must answer before a metered run depends on it:

  1. Does it have room to fall? NEUTRAL did not — it read 1.0 under every
     condition, so no run could support a "binding-specific" claim.
  2. Is it role-blind? Reported as the SIGNED shift against its standard error;
     the absolute value cannot separate a systematic role effect from per-cell
     noise the design already averages away.
  3. Is any profession pair too weak to use? A pair below
     `_MIN_BASELINE_MARGIN` has no discriminability to lose.

CPU-friendly by design (`--model` defaults to a 1.5B) so it can run before
renting a GPU. Re-run on the target model before trusting the numbers; see
docs/CONCEPT_PROBE.md for the recorded 1.5B baseline.

    python scripts/check_concept_probe.py --limit 80
    python scripts/check_concept_probe.py --model Qwen/Qwen3.6-27B --limit 200
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from jspace_binding.experiments.rq2_ablation import _MIN_BASELINE_MARGIN

_EPS = 1e-9


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _load_rows(path: Path, construction: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["construction"] != construction:
                continue
            if not (row.get("concept_probe_entity") and row.get("concept_probe_other")):
                raise SystemExit(
                    "stimuli CSV has no CONCEPT probe columns — regenerate it with "
                    "`python scripts/stimuli_to_csv.py`"
                )
            rows.append(row)
            if len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"no {construction!r} rows in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--stimuli", type=Path, default=Path("data/stimuli/primary_stimuli.csv"))
    parser.add_argument("--construction", default="active_passive")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    # Heavy imports stay local so the module is importable without torch.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = _load_rows(args.stimuli, args.construction, args.limit)

    # token=False: an INVALID stored HF token 401s even public repos, which is
    # far more confusing than having none (RUNBOOK.md section 0). Every repo
    # used here is public.
    tok = AutoTokenizer.from_pretrained(args.model, token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, token=False)
    model.eval()

    def token_id(word: str) -> int | None:
        encoded = tok.encode(" " + word, add_special_tokens=False)
        return encoded[0] if len(encoded) == 1 else None

    concept_margins: list[float] = []
    neutral_margins: list[float] = []
    signed_role_shifts: list[float] = []
    by_pair_margin: dict[str, list[float]] = defaultdict(list)
    by_pair_strict: dict[str, list[bool]] = defaultdict(list)
    multi_token: set[str] = set()

    for row in rows:
        entity, other, counterpart = row["entity"], row["other_entity"], row["counterpart"]
        ids = {w: token_id(w) for w in (entity, other, counterpart)}
        if any(i is None for i in ids.values()):
            multi_token.update(w for w, i in ids.items() if i is None)
            continue
        i_ent, i_oth, i_cp = ids[entity], ids[other], ids[counterpart]
        wanted = (i_ent, i_oth, i_cp)

        def read(sentence: str, probe: str, targets: tuple[int, ...] = wanted) -> dict[int, float]:
            encoded = tok(f"{sentence} {probe}", return_tensors="pt")
            with torch.no_grad():
                dist = torch.softmax(model(**encoded).logits[0, -1, :].float(), dim=-1)
            return {i: float(dist[i]) for i in targets}

        askings = (
            (row["concept_probe_entity"], i_ent, i_oth),
            (row["concept_probe_other"], i_oth, i_ent),
        )

        def concept(
            sentence: str, pairs: tuple[tuple[str, int, int], ...] = askings
        ) -> tuple[float, bool]:
            """Counterbalanced CONCEPT score for one sentence, as rq2 scores it."""
            per_asking, hits = [], []
            for probe, correct, wrong in pairs:
                probs = read(sentence, probe)
                per_asking.append(_logit(probs[correct]) - _logit(probs[wrong]))
                hits.append(probs[correct] > probs[wrong])
            return statistics.mean(per_asking), all(hits)

        scored = {
            cell: concept(row[cell])
            for cell in ("agent_first", "agent_second", "patient_first", "patient_second")
        }
        agent = statistics.mean([scored["agent_first"][0], scored["agent_second"][0]])
        patient = statistics.mean([scored["patient_first"][0], scored["patient_second"][0]])
        signed_role_shifts.append(agent - patient)

        cell_mean = statistics.mean(margin for margin, _ in scored.values())
        concept_margins.append(cell_mean)
        pair = f"{entity}/{other}"
        by_pair_margin[pair].append(cell_mean)
        by_pair_strict[pair].append(all(hit for _, hit in scored.values()))

        # NEUTRAL, the probe being replaced, on the same sentence.
        probs = read(row["agent_first"], row["neutral_probe"])
        neutral_margins.append(
            min(_logit(probs[i_ent]), _logit(probs[i_oth])) - _logit(probs[i_cp])
        )

    n = len(concept_margins)
    print(f"model={args.model}  construction={args.construction}  families={n}")
    if multi_token:
        print(f"skipped (multi-token under this tokenizer): {sorted(multi_token)}")

    concept_mean = statistics.mean(concept_margins)
    print("\n1. HEADROOM  (log-odds margin; same units, so directly comparable)")
    print(f"{'':4s}{'measure':34s} {'mean':>9s} {'median':>9s} {'stdev':>8s}")
    for name, vals in (
        ("NEUTRAL (old control)", neutral_margins),
        ("CONCEPT (new control)", concept_margins),
    ):
        print(
            f"{'':4s}{name:34s} {statistics.mean(vals):+9.3f} "
            f"{statistics.median(vals):+9.3f} {statistics.stdev(vals):8.3f}"
        )
    positive = sum(1 for m in concept_margins if m > 0)
    print(f"{'':4s}CONCEPT margin > 0 on {positive}/{n} families")

    shift = statistics.mean(signed_role_shifts)
    sem = statistics.stdev(signed_role_shifts) / math.sqrt(n)
    systematic = abs(shift) > 2 * sem
    print("\n2. ROLE-BLINDNESS  (signed agent-role minus patient-role margin)")
    print(f"{'':4s}mean signed shift {shift:+.4f}   SEM {sem:.4f}   ratio {abs(shift) / sem:.2f}")
    print(f"{'':4s}{abs(shift) / concept_mean:.1%} of the {concept_mean:+.3f} margin")
    verdict = (
        "SYSTEMATIC: probe tracks role, NOT a valid control"
        if systematic
        else "noise; averaging the four cells removes it"
    )
    print(f"{'':4s}=> {verdict}")

    print(f"\n3. PER-PAIR  (floor _MIN_BASELINE_MARGIN={_MIN_BASELINE_MARGIN})")
    print(f"{'':4s}{'pair':26s} {'n':>3s} {'margin':>9s} {'strict acc':>11s}")
    weak = []
    for pair in sorted(by_pair_margin, key=lambda p: statistics.mean(by_pair_margin[p])):
        margins, strict = by_pair_margin[pair], by_pair_strict[pair]
        mean_margin = statistics.mean(margins)
        print(
            f"{'':4s}{pair:26s} {len(margins):3d} {mean_margin:+9.3f} "
            f"{sum(strict) / len(strict):10.0%}"
        )
        if mean_margin < _MIN_BASELINE_MARGIN:
            weak.append(pair)
    print(
        f"{'':4s}=> {'BELOW FLOOR: ' + ', '.join(weak) if weak else 'all pairs clear the floor'}"
    )
    print("\nstrict acc = all four cells AND both askings correct (8 reads).")


if __name__ == "__main__":
    main()
