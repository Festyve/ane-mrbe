# Runbook: GPU box

One config per model in `configs/`: `default.yaml` (Qwen3.6-27B, layer 48),
`gemma3_12b.yaml`, `gemma3_27b_it.yaml`, `_lre` variants.

## 0. Preconditions

```bash
pip install -e '.[model]'
python scripts/smoke_test.py --model-id Qwen/Qwen2.5-0.5B   # run before spending GPU money
```

- `Qwen/Qwen3.6-27B` + lens: ~80GB disk on a persistent volume.
- 40GB card: `load_in_4bit: true` in the config + `pip install bitsandbytes` (CUDA only).
- Shared box OOM on first forward: set `JSPACE_MAX_MEMORY="0=31GiB,1=15GiB,2=15GiB"`.
- Invalid HF token 401s public repos too (`RepositoryNotFoundError`). Test with:
  `python -c "from huggingface_hub import HfApi; HfApi(token=False).model_info('gpt2')"`

## 1. Run order

```bash
python scripts/replication_gate.py --config configs/default.yaml   # run first
python scripts/run_rq1.py --config configs/default.yaml
python scripts/run_rq2.py --config configs/default.yaml
python scripts/run_rq2.py --config configs/default.yaml --site entity_token

python scripts/fit_directions.py --config configs/default.yaml   # -> data/directions
python scripts/calibrate.py      --config configs/default.yaml   # -> data/calibration.json
python scripts/run_primary.py    --config configs/default.yaml   # loads calibration.json automatically
```

Archive each run: `python scripts/save_run.py <label> --config configs/default.yaml`.

Exit codes: **2** = not configured/fitted yet (checked before the sweep runs).
**3** from `calibrate.py` = no grid value moved behavior — a real outcome, not
a bug. `fit_directions.py --allow-contaminated` overrides the fit/eval-corpus
disjointness check (integration tests only).

## 2. Cost (600 families)

| Stage | Forward passes |
|---|---|
| RQ1 | 4,800 |
| RQ2 | 14,400 / site |

`--site` on `run_rq2.py` halves cost once RQ1 has identified the site.

## 3. Layer band

`model.layer_band` is an inclusive raw-layer pair. `raw = round(reindexed/100 * n_layers)`.
Qwen3.6-27B (64 layers): raw 24–59, mid-band 48. Run band edges as separate
single-layer runs, not one wide band.

## 4. Re-scoring without a GPU

```bash
python scripts/reanalyze_rq2.py --all
python scripts/reanalyze_primary.py --run runs/gemma3-12b-lre --site entity_token
```

Exit 1 if an archived verdict disagrees with the current rule.
