# jspace-binding

Is the **J-space global workspace** a privileged locus for role–filler binding?

A language model that has read *The doctor treated the lawyer* knows two
professions are present and knows which one acted. The J-lens (Gurnee et al.,
2026) exposes a sparse "global workspace" of concepts read out of the residual
stream. This repository tests whether the *binding* — who filled which role —
lives there too, or only the concepts do.

Three lines of evidence, each with a capacity control, across
**Qwen3.6-27B**, **Gemma-3-12B**, and **Gemma-3-27B-IT**:

| | question | method |
|---|---|---|
| **RQ1** | is role *decodable* from J-space? | leave-one-pair-out ridge probes over four activation sources |
| **RQ2** | does the model *use* J-space for binding? | ablate J-space vs a rank-matched random subspace |
| **E3** | can binding be *steered* through J-space? | push a fitted role axis in both signs, against three strength-matched controls |

## The finding

**Role–filler information is present and linearly readable in the residual
stream, but J-space is not a privileged locus for it — J-space carries *less*
role information than an arbitrary subspace of the same rank — and it is not
filler-general.**

Role decodes at 0.989 from the residual stream and 0.587 from J-space, below a
rank-matched random control, and that ordering holds at every layer in the
workspace band and across four orders of magnitude of ridge penalty. Cross-pair
transfer is systematically *inverted* rather than merely absent. Ablating
J-space produces no binding deficit on two of three models; the one positive
causal result (Gemma-3-27B-IT, 0.284 of baseline margin, holding under three
lenses) is confounded between instruction tuning and scale. No model shows a
binding-specific push effect.

**[RESULTS.md](RESULTS.md) is the authoritative record** — every number, every
control, every retraction, and the authoritative artifact for each experiment.
Read it before opening anything under `runs/`.

## Layout

```
src/jspace_binding/    the package: stimuli, directions, interventions, model
                       backends, experiments, analysis
scripts/               one CLI per pipeline stage and per follow-up check
configs/               one YAML per model / variant; paths, layer band, sweep
tests/                 unit + end-to-end contracts, run against DummyModel
runs/                  archived results, one directory per run; the larger
                       ones carry a RUN_INFO.txt saying what they showed, and
                       RESULTS.md indexes which copy is authoritative
docs/ARCHITECTURE.md   data flow, the binding-score math, module contracts
RUNBOOK.md             executing the experiments on a GPU box
```

## Getting started

Everything below runs on a laptop, with no GPU and no model download. The
`DummyModel` backend plants known ground truth in two modes — `binding` (the
pipeline must recover a large positive binding score) and `bag` (it must
recover ~0 inside the control null band) — so the analysis is validated against
known answers before any real weights are loaded.

```bash
pip install -e '.[dev]'
pytest
```

```bash
# End-to-end validation against planted ground truth
python scripts/fit_directions.py --config configs/ci.yaml --dry-run
python scripts/calibrate.py      --config configs/ci.yaml --dry-run
python scripts/run_primary.py    --config configs/ci.yaml --dry-run --dummy-mode binding
python scripts/run_primary.py    --config configs/ci.yaml --dry-run --dummy-mode bag

# The secondary analyses
python scripts/run_rq1.py --config configs/ci.yaml --dry-run   # probe selectivity
python scripts/run_rq2.py --config configs/ci.yaml --dry-run   # ablation deltas
python scripts/run_e4.py  --config configs/ci.yaml --dry-run   # on-demand recruitment
```

To exercise the real backend — model loading, lens loading, sparse pursuit,
edit hooks, token indexing — on a tiny open model with a synthetic
identity-Jacobian lens (needs `pip install -e '.[model]'`):

```bash
python scripts/smoke_test.py --model-id Qwen/Qwen2.5-0.5B
```

For a real run on a GPU box, see [RUNBOOK.md](RUNBOOK.md).

## Reproducing the analysis from the archive

The forward passes are already spent, so every verdict can be recomputed on a
CPU from the archived trials:

```bash
python scripts/reanalyze_rq2.py --all
python scripts/reanalyze_primary.py --run runs/gemma3-12b-lre --site entity_token
```

Both exit 1 if an archived verdict disagrees with the current rule, so they
gate a release rather than being read by eye.

## Reading the results honestly

Five verdict-labelling bugs were found and fixed during analysis, all sharing
one shape: a threshold comparing two quantities without first asking whether
either was distinguishable from chance, from zero, or from its own control. The
measurements were never wrong — only the labels on top of them. Every fix
carries a regression test pinning the observed numbers, and RESULTS.md cites
the commit for each.

Two habits the code enforces as a result:

- **Read `provenance` and `null_interpretable` before trusting a summary.** A
  deficit of zero next to `edit_landed: false` is a plumbing result, not a
  finding.
- **Read per-fold accuracies, not just means.** With as few as three concept
  pairs the mean cannot distinguish "transfers weakly everywhere" from
  "transfers on most pairs and inverts on one".
