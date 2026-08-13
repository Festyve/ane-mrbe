# NLA comparison — scoping note (table #17)

Feasibility and design for comparing our workspace-lens results against
Natural Language Autoencoders on Gemma-3-27B-IT. Written before committing GPU
time, so the blockers are named rather than discovered.

**Bottom line: feasible, and smaller than it first looked.** The integration is
one self-contained script consuming a Parquet file we can already produce. The
real work is a design decision, not engineering: NLA is not a subspace method,
so there is no drop-in "RQ1 with NLA". A new experiment has to be specified.

Sources: [repo](https://github.com/kitft/natural_language_autoencoders),
[`docs/inference.md`](https://raw.githubusercontent.com/kitft/natural_language_autoencoders/main/docs/inference.md),
[checkpoint](https://huggingface.co/kitft/nla-gemma3-27b-L41-av).

---

## 1. Why this model, and what is already resolved

Every released NLA is instruction-tuned. Running `google/gemma-3-27b-it`
therefore removes a confound that would have sunk this comparison on any of our
other models: weights, J-lens, R-lens, and NLA now all refer to the same
checkpoint. That was the reason for choosing the `-it` variant, and it has paid
off — this is the only model where all three readouts are directly comparable.

Available for it:

| readout | artifact |
|---|---|
| J-lens | `neuronpedia/jacobian-lens` · `gemma-3-27b-it/jlens/Salesforce-wikitext` |
| R-lens (+ matched J-lens) | `camilablank/workspace-lenses` · `gemma-3-27b-it/{r-lens,j-lens}` |
| NLA | `kitft/nla-gemma3-27b-L41-{av,ar}` |

## 2. What NLA is, and why it is not a drop-in

Our lens machinery returns a **subspace decomposition** — J-space component,
orthogonal complement, residual — and RQ1 trains probes on those while RQ2
ablates them. NLA returns **English text**: an Activation Verbalizer (AV) reads
one activation vector and emits a description; an Activation Reconstructor (AR)
maps text back to an activation.

So there is no "RQ1 under NLA" in the sense that "RQ1 under R-lens" existed.
The comparison must be a new experiment. The natural one, and the one the PI
proposed: feed the same stimuli, read the verbalization, and ask whether it
recovers the role assignment. "The doctor treated the lawyer" should verbalize
with doctor as agent, not as a bag containing two professions.

This is the load-bearing design decision in the whole task. See §5.

## 3. Mechanics (verified against their docs)

Simpler than feared. Input is a Parquet file with an `activation_vector` column
of `d_model`-wide float lists — which our existing hooks already produce.

Serve the verbalizer:

```bash
python -m sglang.launch_server --model-path kitft/nla-gemma3-27b-L41-av \
    --port 30000 --disable-radix-cache --mem-fraction-static 0.85 \
    --trust-remote-code --attention-backend fa3
```

`--disable-radix-cache` is mandatory: radix caching keys on token IDs, which
are not supplied when injecting embeddings. `--attention-backend fa3` is
required for Gemma-3 (head_dim=256) to avoid OOM.

Then:

```bash
python nla_inference.py kitft/nla-gemma3-27b-L41-av \
    --sglang-url http://localhost:30000 --parquet activations.parquet
```

Output is `<explanation>...</explanation>`, 2–3 natural-language snippets per
activation, at roughly 2–5 s per request on an A100/H100.

### The two scale factors — non-negotiable

```
v_scaled = v_raw * (injection_scale / ||v_raw||_fp32)
embeds   = embed_layer(input_ids) * embed_scale
```

For Gemma-3-27B: `injection_scale = 60000`, `embed_scale = sqrt(5376) ~= 73`.
Gemma-3's forward pass multiplies embeddings by `sqrt(hidden_size)`, and that
step is bypassed when loading raw weights. Their docs call omitting it
"near-garbage output", and give a memorable failure signature: **output in
Chinese across all inputs means injection failed** — check these two numbers
first.

Both must be read from the checkpoint's `nla_meta.yaml` sidecar, never
hardcoded.

## 4. Blockers, in order of severity

**1. Memory. Their docs say 27B needs 80 GB; our cards are 40 GB.** This is the
hard one. Options: shard SGLang across two cards (extra setup, untested by us),
or run the comparison on Gemma-3-12B instead (`kitft/nla-gemma3-12b-L32-av`,
`injection_scale = 80000`, `embed_scale = sqrt(3840) ~= 62`) and accept that
the three-way tool comparison then spans two models rather than one.

**2. Layer mismatch. The NLA reads block 41 of 62; we read 46.** Our own layer
sweep showed selectivity varies materially across the band (the
J-space-minus-random difference flips sign between L24 and L57), so this is not
negligible. Cheapest fix is to extract a second set of activations at 41; that
is a read-only pass, no refitting.

**3. Site anchoring.** Our reads are anchored to a sentence position (entity
token or final token). The NLA expects one vector per row with no notion of
which position it came from. Mapping is mechanical but must be recorded, since
"which token" is exactly the variable our site dissociation turns on.

## 5. Scoring — the actual design problem

Judging whether a verbalization "got the binding right" is not obvious, and
whatever we choose must be pre-registered. This project has already found five
verdict labels that claimed more than their measurement supported; inventing a
sixth by tuning a rubric after seeing outputs would be worse than not running
the experiment.

**Option A — AR reconstruction MSE (quantitative, recommended).** The AR
checkpoint scores decode fidelity numerically: normalize predicted and gold
vectors, take MSE (equal to `2(1 - cos)`), where ~0.2 is good reconstruction and
~2.0 is orthogonal. Crucially this supports our **minimal-pair** design: run the
agent-role and patient-role cells of the same family and ask whether their
verbalizations reconstruct to *different* activations. If the NLA is blind to
role, the two texts collapse to the same reconstruction. This needs no rubric
and no judge, and inherits the pairing our stimuli already have.

**Option B — string presence.** Cheap, and too weak: both professions appear in
both cells, so "doctor" and "lawyer" being present says nothing about who did
what.

**Option C — LLM judge.** Flexible, but introduces a whole new instrument that
would itself need validating, on a project whose central finding is that
verdicts must be checked against controls.

**Recommendation: Option A as primary, with verbalizations reported
qualitatively alongside.** The MSE is the pre-registerable number; the text is
what makes the paper's point legible to a reader.

**Controls that must come along**, matching how every other experiment here is
gated: the same minimal-pair contrast under a random activation, and under an
activation from a non-participant entity. Without them this reproduces exactly
the failure mode the primary just corrected — a real-looking effect that any
strength-matched input would produce.

## 6. Estimated cost

| step | cost | GPU |
|---|---|---|
| Read API, write the Parquet exporter | half a day | no |
| Fix the memory blocker (§4.1) | half to one day | no |
| Extract activations at L41 | ~30 min | yes |
| SGLang setup + smoke test on a few rows | half a day | yes |
| Full run, 600 families × 2 role cells | 2–5 s/request → ~1–2 h | yes |
| AR scoring + controls | half a day | yes |

**Roughly 2–3 days**, of which the first half is GPU-free and can start now.
The estimate is dominated by the memory blocker and the smoke test, not by the
run itself.

## 7. Recommendation

Start the GPU-free half now: write the Parquet exporter, read `nla_meta.yaml`
for the real scale factors, and settle the scoring design. That surfaces
whether §4.1 is a half-day or a wall before any card is committed.

If the 80 GB requirement proves immovable, running NLA on **Gemma-3-12B** is a
reasonable fallback: we have the full pipeline on that model already, and the
comparison becomes "NLA vs J-lens on 12B" plus "R-lens vs J-lens on 27B-IT"
rather than three tools on one model. Weaker, but real, and it beats deferring
the whole thing.
