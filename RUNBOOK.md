# Runbook: executing the experiments on a GPU box

Ordered steps, the manual gotchas, and what has actually been verified against
the real artifacts versus what has only ever run against `DummyModel`.

## 0. Preconditions

**Hardware.** `Qwen/Qwen3.6-27B` is 55.6 GB and the lens is 3.3 GB, so budget
**~80 GB of disk** and put it on a persistent volume — otherwise every session
re-downloads 59 GB. An A100 80GB / H100 runs bf16 directly. On a 40 GB card use
`load_in_4bit: true` and `pip install bitsandbytes` (CUDA only — it does not
build on Apple Silicon, so none of this runs on a Mac).

`device_map: auto` is correct on CUDA. The "segfaults" note in
`QwenJLensModel._load_model` is CPU-only-Mac specific.

**HuggingFace auth.** All four repos are public and ungated, so a token is only
needed for rate limits and download speed. An *invalid* token is worse than
none: it 401s every request, including public ones. If downloads fail with
`RepositoryNotFoundError`, test whether anonymous access works before assuming
the repo moved:

```python
from huggingface_hub import HfApi
HfApi(token=False).model_info("gpt2")   # succeeds => your token is the problem
```

**Extras.** `pip install -e '.[model]'` for torch/transformers/accelerate.

## 1. Run order

RQ1 and RQ2 need only the model and the lens. The primary experiment needs two
artifact-producing stages first, and a manual config edit between them.

```bash
# --- no prerequisites beyond model + lens -------------------------------
python scripts/run_rq1.py --config configs/default.yaml
python scripts/run_rq2.py --config configs/default.yaml          # both sites
python scripts/run_rq2.py --config configs/default.yaml --site entity_token

# --- primary experiment pipeline ---------------------------------------
python scripts/fit_directions.py --config configs/default.yaml   # -> data/directions
python scripts/calibrate.py      --config configs/default.yaml   # -> alpha, push_coefficient

#    !! MANUAL !! copy the two printed values into configs/default.yaml as
#    model.alpha and model.push_coefficient. The runner reads them from CONFIG;
#    data/calibration.json is only the audit record. Skipping this silently
#    runs the primary experiment with a pure swap and no push scaling.

python scripts/run_primary.py --config configs/default.yaml
```

`run_primary.py` calls `preflight_or_exit(model, injection_sites)`, which
requires fitted directions and **exits 2** before the sweep if they are absent.
That check is deliberate: a mid-sweep failure is then a real error rather than a
setup problem.

### Two failure modes that are outcomes, not bugs

- `calibrate.py` **exit 3** — no grid value moved behavior enough. This is the
  proposal's uninterpretable-null outcome. Budget for the possibility that the
  pipeline stops here.
- `fit_directions.py` refuses a `--corpus` whose sentences occur verbatim in the
  primary stimuli (fitting and testing on the same sentences contaminates the
  causal test). `--allow-contaminated` overrides and stamps
  `"contaminated": true` into the summary — throwaway integration tests only.

## 2. Cost

At the default 600 families:

| Stage | Forward passes |
|---|---|
| RQ1 | 600 x 4 cells x 2 sites = **4,800** |
| RQ2 | 600 x 3 conditions x 4 cells x 2 probes x **per site** = **14,400/site** |

RQ2 defaults to every configured site because ablating where RQ1 found no signal
and reporting no effect is close to tautological. Use `--site` to halve it once
RQ1 has identified the site that carries the signal.

## 3. Verified against the real artifacts

Checked without a GPU; these should not need rediscovering:

- `Qwen/Qwen3.6-27B` exists, **64 layers, hidden_size 5120**, 55.6 GB / 29 files.
- Despite the multimodal `Qwen3_5ForConditionalGeneration` architecture,
  `AutoModelForCausalLM` resolves to the text tower `Qwen3_5ForCausalLM`, so
  `_decoder_layer`'s `model.model.layers` is correct (64 layers) and the
  unembedding is (248320, 5120).
- Lens subpath `qwen3.6-27b/jlens/Salesforce-wikitext` resolves to a real
  3.30 GB `.pt`. Its pickle carries exactly the documented schema —
  `{J, source_layers, d_model, n_prompts}`, fp16 storages, 65 layer tensors of
  5120x5120 — matching `_JACOBIAN_KEY_PATTERNS` / `_read_arrays`.
- All 12 answer-vocabulary words are single-token under the real tokenizer
  (`_single_token_id` raises otherwise), and the tokenizer is fast, which
  `_last_word_token_index` needs for `offset_mapping`.
- All 4,800 `entity_token` index lookups over the real stimuli resolve.
- `configs/default.yaml` has no open decisions for RQ1/RQ2: `missing_decisions`
  is empty and `layer_band` is `(48, 48)`.

## 4. NOT yet verified — expect first-run friction

`fit_directions.py`, `calibrate.py`, and `run_primary.py` have **never executed
against the real backend**. Their test coverage is against `DummyModel` only.
The forward/hook path is shared with RQ1/RQ2 so the risk is concentrated in the
direction-fitting and calibration logic rather than in model plumbing, but do
the first run interactively rather than as a fire-and-forget batch job.

Local smoke test before spending GPU money — real weights, tiny model, exercises
every backend stage:

```bash
python scripts/smoke_test.py --model-id Qwen/Qwen2.5-0.5B
```

## 5. Known-degenerate results (code cannot fix these)

Running successfully does not make every number interpretable. Four design
issues are live, and the summaries now flag rather than hide them:

1. **3 concept pairs => 3 folds.** `leave_one_pair_out` averages very few
   numbers; one pair transferring backwards drags the mean below chance, which
   reads as "no signal" but is a transfer failure. Check `fold_accuracies` and
   `inverting_folds` in `rq1_probe.json` before interpreting any mean.
2. **`orthogonal` is ~99.7% of `residual` by construction** (jspace is 16 of
   5120 dims), so that contrast cannot localize anything.
3. **RQ1 has no capacity control.** jspace has effective rank <= 16 while
   `orthogonal` has ~5120, so comparing their accuracies conflates localization
   with capacity. RQ2 has the matched `ABLATE_RANDOM_SUBSPACE` control; RQ1 has
   no equivalent.
4. **The recall control sits at ceiling** — exactly 1.0 under every condition,
   in the dummy as well as on the real model. It therefore cannot register
   damage, and the binding-minus-recall subtraction reduces to the raw binding
   deficit. **No run using the current neutral probe can support a
   "binding-specific" claim.** `recall_control_informative` reports this per
   site, and `null_interpretable` folds it together with the edit-magnitude
   check.

Always read `provenance` (model, lens, layer band, commit, dirty flag) and
`null_interpretable` before trusting a summary. A deficit of zero next to
`edit_landed: false` is a plumbing result, not a finding.
