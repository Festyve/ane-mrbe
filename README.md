#  Bag of Concepts? Testing Role-Filler Binding in the Verbalizable Workspace

Code, data, and analysis scripts for *A Bag of Concepts? Testing Role-Filler
Binding in the Verbalizable Workspace*.

[RESULTS.md](RESULTS.md): numeric record. [RUNBOOK.md](RUNBOOK.md): GPU run
procedure.

## Install

```bash
pip install -e '.[dev]'      # DummyModel only, no GPU
pip install -e '.[model]'    # + real qwen_jlens backend
```

## Verify install

```bash
pytest

python scripts/fit_directions.py --config configs/ci.yaml --dry-run
python scripts/calibrate.py      --config configs/ci.yaml --dry-run
python scripts/run_primary.py    --config configs/ci.yaml --dry-run --dummy-mode binding
python scripts/run_primary.py    --config configs/ci.yaml --dry-run --dummy-mode bag
python scripts/run_rq1.py        --config configs/ci.yaml --dry-run
python scripts/run_rq2.py        --config configs/ci.yaml --dry-run

python scripts/smoke_test.py --model-id Qwen/Qwen2.5-0.5B   # real backend, tiny model
```

## Reproduce a paper result

Configs: `configs/default.yaml` (Qwen3.6-27B), `configs/gemma3_12b.yaml`,
`configs/gemma3_27b_it.yaml`, `_lre` variants, `configs/gemma3_27b_it_jlens_matched.yaml`,
`configs/gemma3_27b_it_rlens.yaml`. Run `fit_directions.py` then `calibrate.py`
against a config before any script below that needs directions/calibration —
order and GPU cost in [RUNBOOK.md](RUNBOOK.md).

| Result | Command | Output |
|---|---|---|
| Table 1 — within-pair decoding | `python scripts/check_within_pair.py --config configs/default.yaml --site final_token` | `data/results/within_pair.json` |
| §5.1 within-pair, entity token | `python scripts/check_within_pair.py --config configs/default.yaml --site entity_token` | `data/results/within_pair.json` |
| Appendix B — nonlinear probes | `python scripts/check_nonlinear_probe.py --config configs/default.yaml` | `data/results/nonlinear_probe.json` |
| Appendix B — ridge-penalty sweep | `python scripts/check_ridge_penalty.py --config configs/default.yaml` | `data/results/ridge_penalty_sweep.json` |
| Appendix B — layer sweep | `python scripts/run_rq1.py --config configs/default.yaml --layer 24` (and `59`) | `data/results_L{24,59}/rq1_probe.json` |
| Appendix B — seed replication | `python scripts/run_rq1.py --config configs/default.yaml --seed 42` | `data/results_S42/rq1_probe.json` |
| Appendix C — cross-pair transfer, 3 pairs | `python scripts/run_rq1.py --config configs/default.yaml` | `data/results/rq1_probe.json` |
| §5.1 cross-pair transfer, 6 pairs | `python scripts/run_rq1.py --config configs/expanded_pairs.yaml` | `data/results_6pair/rq1_probe.json` |
| §4 replication gate | `python scripts/replication_gate.py --config configs/default.yaml` | stdout |
| §3 direction fitting | `python scripts/fit_directions.py --config configs/default.yaml` | `data/directions/directions_{site}.npz` |
| §3 calibration | `python scripts/calibrate.py --config configs/default.yaml --site entity_token` | `data/calibration.json` |
| §5.2 Table 2 — LRE push sweep (Gemma-3-27B-IT) | `python scripts/run_primary.py --config configs/gemma3_27b_it_lre.yaml --site entity_token` | stdout |
| §5.2 primary push / E3 | `python scripts/run_primary.py --config configs/default.yaml --site entity_token` | stdout (no summary file) |
| §5.3 Table 3 — RQ2 ablation | `python scripts/run_rq2.py --config configs/default.yaml --site entity_token` (and `final_token`) | `data/results/rq2_ablation.json` |
| Appendix D — 3-lens replication | `python scripts/run_rq2.py --config configs/gemma3_27b_it_jlens_matched.yaml --site entity_token`, then `--config configs/gemma3_27b_it_rlens.yaml` | `data/results/rq2_ablation.json` |

## Re-score archived verdicts (no GPU)

```bash
python scripts/reanalyze_rq2.py --all
python scripts/reanalyze_primary.py --run runs/gemma3-12b-lre --site entity_token
```

## Layout

```
src/jspace_binding/    stimuli, direction fitting, interventions, model backends, experiments, analysis
scripts/               one CLI per pipeline stage / robustness check
configs/               one YAML per model / lens / estimator variant
tests/                 unit + end-to-end contracts, run against DummyModel
runs/                  archived results (RESULTS.md indexes the authoritative copy)
docs/                  pointers into src/ — see below
```

## License

MIT — see [LICENSE](LICENSE).
