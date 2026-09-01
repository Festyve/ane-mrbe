# Runbook: executing the experiments on a GPU box

Ordered steps and the manual gotchas. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
covers the layout; [RESULTS.md](RESULTS.md) covers what the runs found.

One config per model lives in `configs/`. The examples below use
`configs/default.yaml` (Qwen3.6-27B, layer 48); substitute
`gemma3_12b.yaml` or `gemma3_27b_it.yaml` for the others, and the `_lre`
variants for the gradient estimator.

## 0. Preconditions

**Hardware.** `Qwen/Qwen3.6-27B` is 55.6 GB and its lens is 3.3 GB, so budget
**~80 GB of disk on a persistent volume** — otherwise every session
re-downloads 59 GB. An A100 80GB / H100 runs bf16 directly. On a 40 GB card use
`load_in_4bit: true` and `pip install bitsandbytes` (CUDA only; it does not
build on Apple Silicon, so none of this runs on a Mac).

`device_map: auto` is correct on CUDA. On a shared box where "auto" packs the
first card to the brim and then OOMs at the first forward, set
`JSPACE_MAX_MEMORY="0=31GiB,1=15GiB,2=15GiB"` (visible indices) to reserve
headroom.

**HuggingFace auth.** All four repos are public and ungated, so a token is only
needed for rate limits and download speed. An *invalid* token is worse than
none: it 401s every request, including public ones, and the error reads
`RepositoryNotFoundError`, which looks like a wrong model id. Test anonymous
access before assuming the repo moved:

```python
from huggingface_hub import HfApi
HfApi(token=False).model_info("gpt2")   # succeeds => your token is the problem
```

**Extras.** `pip install -e '.[model]'` for torch/transformers/accelerate.

**Smoke test first.** `python scripts/smoke_test.py --model-id Qwen/Qwen2.5-0.5B`
exercises every backend stage on real weights with a synthetic identity lens.
Run it before spending GPU money.

## 1. Run order

RQ1 and RQ2 need only the model and the lens. The primary experiment needs
two artifact-producing stages first, and a manual config edit between them.

```bash
# --- no prerequisites beyond model + lens -------------------------------
python scripts/replication_gate.py --config configs/default.yaml   # run this first
python scripts/run_rq1.py --config configs/default.yaml
python scripts/run_rq2.py --config configs/default.yaml            # both sites
python scripts/run_rq2.py --config configs/default.yaml --site entity_token

# --- primary experiment pipeline ---------------------------------------
python scripts/fit_directions.py --config configs/default.yaml   # -> data/directions
python scripts/calibrate.py      --config configs/default.yaml   # -> data/calibration.json

#    run_primary.py now loads model.alpha and model.push_coefficient from
#    data/calibration.json automatically, matching on the calibrated site. No
#    manual YAML edit is needed. Set model.push_coefficient / model.alpha in the
#    config only to OVERRIDE the record. If neither the config nor a matching
#    calibration supplies a push coefficient, run_primary.py now EXITS 2 before
#    the sweep instead of silently running a pure swap with no push scaling.

python scripts/run_primary.py --config configs/default.yaml
```

Archive each run as you go — `python scripts/save_run.py <label>` copies the
config's output paths into `runs/` and pushes them, which matters on an
ephemeral box where `.gitignore` excludes every output location.

**Run the replication gate first.** When an experiment reports "role is not in
J-space", that is indistinguishable from "the lens is not reading" from its own
output — both produce near-chance numbers, and they call for opposite
responses. The gate is five minutes and separates them.

### Exit codes that are outcomes, not bugs

- **exit 2** — the backend is not runnable yet: a missing config decision, lens
  artifact, or unfitted direction. `preflight_or_exit` checks this *before* the
  sweep, so a mid-sweep failure is always a real error.
- **exit 3** from `calibrate.py` — no grid value moved behavior enough. This is
  the uninterpretable-null outcome, and it is where the Qwen pipeline actually
  stopped. Budget for it.
- `fit_directions.py` refuses a `--corpus` whose sentences occur verbatim in the
  primary stimuli, since fitting and testing on the same sentences contaminates
  the causal test. `--allow-contaminated` overrides and stamps
  `"contaminated": true` into the summary — throwaway integration tests only.

## 2. Cost

At the default 600 families:

| Stage | Forward passes |
|---|---|
| RQ1 | 600 x 4 cells x 2 sites = **4,800** |
| RQ2 | 600 x 3 conditions x 4 cells x 2 probes **per site** = **14,400/site** |

RQ2 defaults to every configured site, because ablating where RQ1 found no
signal and reporting no effect is close to tautological. Use `--site` to halve
it once RQ1 has identified the site carrying the signal.

## 3. Pinning the layer band

`model.layer_band` is an inclusive pair of **raw** layer indices. The source
paper's workspace is reindexed layers ~38–92 of 100, with single-layer analyses
mid-band (~L75 reindexed); convert with `raw = round(reindexed/100 * n_layers)`.
For Qwen3.6-27B (64 layers, d_model 5120) that is raw 24–59, mid-band ~48.

Run the band edges as separate **single-layer** runs, not as one wide band. A
`[24, 59]` band edits all 36 layers at once — ~36x the per-trial edit cost, and
it cannot localise anything, since one smeared edit cannot distinguish "binding
lives at L48" from "binding is everywhere".

## 4. Known-degenerate results the code cannot fix

Running successfully does not make every number interpretable. Four design
issues are live, and the summaries flag rather than hide them.

1. **3 concept pairs => 3 folds.** `leave_one_pair_out` averages very few
   numbers, and one pair transferring backwards drags the mean below chance,
   which reads as "no signal" but is a transfer failure. Check
   `fold_accuracies` and `inverting_folds` in `rq1_probe.json` before
   interpreting any mean. `configs/expanded_pairs.yaml` raises this to 6.
2. **`orthogonal` is ~99.7% of `residual` by construction** (jspace is 16 of
   5120 dims), so that contrast cannot localize anything.
3. **RQ1 has no capacity-matched *ablation* control.** It has
   `random_subspace`, which matches rank, but RQ2's `ABLATE_RANDOM_SUBSPACE`
   matches rank and not perturbation magnitude — the two differ by ~5x, because
   J-space ablation removes the most *active* directions. A J-space effect
   exceeding the control is therefore not by itself evidence of localisation.
4. **The old NEUTRAL recall control sat at ceiling** — exactly 1.0 under every
   condition, so it could not register damage and the binding-minus-recall
   subtraction reduced to the raw binding deficit. RQ2 now uses the
   counterbalanced CONCEPT probe instead (docs/CONCEPT_PROBE.md);
   `recall_control_informative` reports whether it did its job on a given run,
   and `null_interpretable` folds that together with the edit-magnitude check.

Always read `provenance` (model, lens, layer band, commit, dirty flag) and
`null_interpretable` before trusting a summary. **A deficit of zero next to
`edit_landed: false` is a plumbing result, not a finding.**

## 5. Re-scoring without a GPU

The forward passes are spent, so archived runs can be re-scored on a laptop
whenever a verdict rule changes:

```bash
python scripts/reanalyze_rq2.py --all
python scripts/reanalyze_primary.py --run runs/gemma3-12b-lre --site entity_token
```

Both exit 1 if an archived verdict disagrees with the current rule. Neither
recomputes a measurement.
