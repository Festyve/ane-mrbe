# Role-direction sanity check

**Status: synthetic-backend results only.** Every number below comes from the
GPU-free `DummyModel` (`--dry-run`), whose activations are generated with a
*planted* role axis (`binding` mode) or as pure noise (`bag` mode). They
validate the harness — fit → held-out separation → comparison — not any claim
about Qwen. Rerun both commands without `--dry-run` once the real backend is
runnable (layer band pinned, working arm64 torch) and replace this section's
numbers before circulating.

Reproduce:

```bash
python scripts/fit_directions.py   --dry-run --dummy-mode binding \
    --entities doctor,nurse,teacher,driver,chef,farmer,coach
python scripts/direction_sanity.py --dry-run --dummy-mode binding --entity doctor,nurse
# negative control:
python scripts/fit_directions.py   --dry-run --dummy-mode bag --entities doctor,nurse
python scripts/direction_sanity.py --dry-run --dummy-mode bag --entity doctor,nurse
```

## What the check does

This complements — not replaces — the proposal's Datasets §1 pilot check (the
bootstrap-stability resample, run at fit time by `scripts/fit_directions.py`):
stability asks whether the fitting corpus pins down *a* direction; the checks
below ask whether that direction *generalizes* and how the per-entity
directions relate.

1. **Held-out separation.** Directions are fit on the fitting corpus only
   (24 exemplars per role per entity). The check projects *held-out* sentences
   — either the hand-written `data/handwritten/eval_doctor.jsonl` or generated
   frames verified verbatim-disjoint from the fitting set — onto each entity's
   fitted unit direction, and scores the 1-D split: Mann-Whitney AUC
   (threshold-free), accuracy at the class-mean midpoint (the "2-line
   classifier"), and Cohen's d. A strip-scatter per site lands in
   `figures/direction_sanity_{site}.png` where matplotlib is available.
2. **Doctor vs nurse comparison.** Pairwise cosine between fitted unit
   directions at each injection site, bucketed into near-identical
   (filler-general) / related-but-distinct / distinct.

## Results (synthetic)

### Held-out separation — positive signal (`binding` mode)

| entity | site | held-out n | AUC | accuracy | Cohen's d |
|---|---|---|---|---|---|
| doctor | final_token | 24 + 24 | 1.000 | 1.000 | 5.7 |
| nurse | final_token | 24 + 24 | 1.000 | 1.000 | 6.3 |
| doctor | entity_token | 24 + 24 | 1.000 | 1.000 | 7.2 |
| nurse | entity_token | 24 + 24 | 1.000 | 1.000 | 5.9 |
| doctor (hand-written eval set) | final_token | 16 + 16 | 1.000 | 1.000 | 5.5 |

The fitted directions separate held-out agent from patient activations
perfectly — as they must, since binding mode plants exactly such an axis. The
value of the run is that the *pipeline* recovers it out-of-sample: fit on the
training frames only, separation measured on unseen verb/distractor
combinations and on the hand-written eval set.

### Negative control (`bag` mode)

| entity | site | AUC | accuracy | Cohen's d |
|---|---|---|---|---|
| doctor | final_token | 0.495 | 0.500 | 0.18 |
| nurse | final_token | 0.491 | 0.500 | 0.05 |
| doctor | entity_token | 0.458 | 0.521 | 0.10 |
| nurse | entity_token | 0.387 | 0.562 | 0.34 |

With no role information in the activations, AUC sits at chance and the fit
script's bootstrap-stability warning fires for every direction. This is the
check working: it cannot be fooled into reporting separation that is not
there, and a direction fit on noise is flagged *before* anything downstream
consumes it.

### Doctor vs nurse directions

Mean off-diagonal cosine ≈ **0.00** at both sites → "essentially distinct" on
the synthetic backend. **Do not read anything into this**: the DummyModel
plants an independent random axis per entity, so near-zero cosine is its
ground truth by construction. On the real model this comparison is the
substantive question — Feng & Steinhardt-style filler-general binding predicts
high cosine (one shared role axis), entity-specific binding predicts low. The
harness distinguishes the three regimes (the verdict buckets are exercised in
`tests/test_direction_sanity.py`); the answer awaits the real backend.

## Interpretation guardrails

- Linear separation on held-out sentences shows the direction *carries* role
  information out-of-sample — a precondition for the causal push experiments,
  not evidence the model *uses* it (that is RQ2's ablation).
- The held-out generated frames share templates with the fitting frames
  (unseen verb/distractor combinations, same syntax). The hand-written eval
  set is the stronger held-out test; extend it beyond doctor when scaling up.
- AUC = 1.0 will not survive contact with the real model; decide in advance
  what counts as passing (the `separates` flag uses AUC ≥ 0.75, a heuristic
  worth revisiting against the pilot data).
