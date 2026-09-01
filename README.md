# jspace-binding

Code, data, and analysis scripts for *A Bag of Concepts? Testing Role-Filler
Binding in the Verbalizable Workspace* (paper.pdf). Tests whether role-filler
binding is decodable from, steerable through, and causally load-bearing in the
J-lens/R-lens workspace of Qwen3.6-27B, Gemma-3-12B, and Gemma-3-27B-IT.

[RESULTS.md](RESULTS.md) is the authoritative numeric record; this file only
maps paper results to the command that reproduces each one.

## Install

```bash
pip install -e '.[dev]'      # numpy/matplotlib/pyyaml + pytest/ruff — runs everything on DummyModel
pip install -e '.[model]'    # + torch/transformers/accelerate/hf_hub, for the real qwen_jlens backend
```

## Verify the install (no GPU, no model download)

```bash
pytest

python scripts/fit_directions.py --config configs/ci.yaml --dry-run
python scripts/calibrate.py      --config configs/ci.yaml --dry-run
python scripts/run_primary.py    --config configs/ci.yaml --dry-run --dummy-mode binding
python scripts/run_primary.py    --config configs/ci.yaml --dry-run --dummy-mode bag
python scripts/run_rq1.py        --config configs/ci.yaml --dry-run
python scripts/run_rq2.py        --config configs/ci.yaml --dry-run
```

To exercise the real backend (model loading, lens loading, sparse pursuit,
edit hooks) on a tiny open model with a synthetic identity lens:

```bash
python scripts/smoke_test.py --model-id Qwen/Qwen2.5-0.5B
```

For a real run on a GPU box: [RUNBOOK.md](RUNBOOK.md).

## Reproducing each paper result

Configs: `configs/default.yaml` (Qwen3.6-27B), `configs/gemma3_12b.yaml`,
`configs/gemma3_27b_it.yaml` (primary J-lens), plus `_lre` variants (LRE
gradient estimator in place of difference-of-means) and, for Gemma-3-27B-IT,
`configs/gemma3_27b_it_jlens_matched.yaml` / `configs/gemma3_27b_it_rlens.yaml`.
Every real run needs `scripts/fit_directions.py` and `scripts/calibrate.py`
run first against the same config — see [RUNBOOK.md](RUNBOOK.md) for order and
GPU cost. Output paths below are `config.paths.*` for `configs/default.yaml`;
other configs redirect under `data/<model>/`.

| Paper result | Command | Output |
|---|---|---|
| Table 1 — within-pair decoding (sentence-final) | `python scripts/check_within_pair.py --config configs/default.yaml --site final_token` (repeat per model config) | `data/results/within_pair.json` |
| §5.1 within-pair at entity token | `python scripts/check_within_pair.py --config configs/default.yaml --site entity_token` | `data/results/within_pair.json` |
| Appendix B — nonlinear probes (quad/RFF) | `python scripts/check_nonlinear_probe.py --config configs/default.yaml` | `data/results/nonlinear_probe.json` |
| Appendix B — ridge-penalty sweep | `python scripts/check_ridge_penalty.py --config configs/default.yaml` | `data/results/ridge_penalty_sweep.json` |
| Appendix B — layer sweep (band edges) | `python scripts/run_rq1.py --config configs/default.yaml --layer 24` (and `59`) | `data/results_L{24,59}/rq1_probe.json` |
| Appendix B — seed replication | `python scripts/run_rq1.py --config configs/default.yaml --seed 42` | `data/results_S42/rq1_probe.json` |
| Appendix C — cross-pair transfer, 3 pairs | `python scripts/run_rq1.py --config configs/default.yaml` | `data/results/rq1_probe.json` |
| §5.1 cross-pair transfer, 6 pairs | `python scripts/run_rq1.py --config configs/expanded_pairs.yaml` | `data/results_6pair/rq1_probe.json` |
| §4 replication gate | `python scripts/replication_gate.py --config configs/default.yaml` | stdout (no file) |
| §3 direction fitting + stability | `python scripts/fit_directions.py --config configs/default.yaml` | `data/directions/directions_{site}.npz` |
| §3 push / identity-swap calibration | `python scripts/calibrate.py --config configs/default.yaml --site entity_token` | `data/calibration.json` |
| §5.2 Table 2 — LRE push-coefficient sweep (Gemma-3-27B-IT) | `python scripts/run_primary.py --config configs/gemma3_27b_it_lre.yaml --site entity_token` (c set in the config; sweep runs at 64/128/256) | stdout (see below) |
| §5.2 primary push / E3 (other models) | `python scripts/run_primary.py --config configs/default.yaml --site entity_token` | stdout only — `run_primary.py` prints its analysis and writes no summary file; redirect to a log |
| §5.3 Table 3 — RQ2 ablation | `python scripts/run_rq2.py --config configs/default.yaml --site entity_token` (and `final_token`) | `data/results/rq2_ablation.json` |
| Appendix D — 3-lens replication (Gemma-3-27B-IT) | `python scripts/run_rq2.py --config configs/gemma3_27b_it_jlens_matched.yaml --site entity_token` and `--config configs/gemma3_27b_it_rlens.yaml` | `data/results/rq2_ablation.json` per config |

## Re-scoring archived verdicts (no GPU)

The forward passes for every number above are already spent and archived
under `runs/`; re-deriving a verdict from the raw trials needs no model:

```bash
python scripts/reanalyze_rq2.py --all
python scripts/reanalyze_primary.py --run runs/gemma3-12b-lre --site entity_token
```

Both exit 1 if an archived verdict disagrees with the current scoring rule.

## Layout

```
src/jspace_binding/    stimuli, direction fitting, interventions, model backends, experiments, analysis
scripts/               one CLI per pipeline stage and per robustness check
configs/               one YAML per model / lens / estimator variant
tests/                 unit + end-to-end contracts, run against DummyModel
runs/                  archived results, one directory per run (RESULTS.md indexes the authoritative copy)
docs/                  ARCHITECTURE.md, CONCEPT_PROBE.md — pointers, see below
RESULTS.md             every number, every control — the authoritative record
RUNBOOK.md             executing the experiments on a GPU box
```

## License

MIT — see [LICENSE](LICENSE).
