# NLA comparison — design note, not run

**Status: incomplete.** The activation export exists
(`runs/gemma3-27b-it-nla/activations_L41.parquet`, 2400 rows) and
`scripts/export_nla_activations.py` produces it, but the verbaliser never ran:
SGLang shards the 108 GB checkpoint across two 40 GB cards fine, and the blocker
was card availability on a box with all 8 GPUs in `Exclusive_Process` mode
(`cudaErrorDevicesUnavailable` at init), not memory. This note records the
design so the export is interpretable and the run is resumable.

Sources: [repo](https://github.com/kitft/natural_language_autoencoders),
[checkpoint](https://huggingface.co/kitft/nla-gemma3-27b-L41-av).

## Why Gemma-3-27B-IT

Every released NLA is instruction-tuned, so running `google/gemma-3-27b-it`
makes weights, J-lens, R-lens, and NLA all refer to the same checkpoint. It is
the only model in this project where all three readouts are directly
comparable.

| readout | artifact |
|---|---|
| J-lens | `neuronpedia/jacobian-lens` · `gemma-3-27b-it/jlens/Salesforce-wikitext` |
| R-lens (+ matched J-lens) | `camilablank/workspace-lenses` · `gemma-3-27b-it/{r-lens,j-lens}` |
| NLA | `kitft/nla-gemma3-27b-L41-{av,ar}` |

## Why it is not a drop-in

The lens machinery returns a **subspace decomposition** — J-space component,
orthogonal complement, residual — which RQ1 probes and RQ2 ablates. NLA returns
**English text**: an Activation Verbalizer reads one activation vector and emits
a description, and an Activation Reconstructor maps text back to an activation.

So there is no "RQ1 under NLA" the way "RQ1 under R-lens" existed. The
comparison has to be a new experiment: feed the same stimuli, read the
verbalization, and ask whether it recovers the role assignment. *The doctor
treated the lawyer* should verbalize with doctor as agent, not as a bag
containing two professions.

## Mechanics

Input is a Parquet with an `activation_vector` column of `d_model`-wide float
lists, which the existing hooks already produce.

```bash
python -m sglang.launch_server --model-path kitft/nla-gemma3-27b-L41-av \
    --port 30000 --disable-radix-cache --mem-fraction-static 0.85 \
    --trust-remote-code --attention-backend fa3

python nla_inference.py kitft/nla-gemma3-27b-L41-av \
    --sglang-url http://localhost:30000 --parquet activations.parquet
```

`--disable-radix-cache` is mandatory — radix caching keys on token IDs, which
are not supplied when injecting embeddings. `--attention-backend fa3` is
required for Gemma-3 (head_dim=256) to avoid OOM. Output is
`<explanation>...</explanation>`, 2–3 snippets per activation, ~2–5 s per
request on an A100/H100.

### The two scale factors

```
v_scaled = v_raw * (injection_scale / ||v_raw||_fp32)
embeds   = embed_layer(input_ids) * embed_scale
```

For Gemma-3-27B, `injection_scale = 60000` and `embed_scale = sqrt(5376) ≈ 73`.
Gemma-3's forward pass multiplies embeddings by `sqrt(hidden_size)`, and that
step is bypassed when loading raw weights. Both must be read from the
checkpoint's `nla_meta.yaml`, never hardcoded, and the export must stay
unscaled — applying the rescale twice produces the documented failure
signature, **output in Chinese across all inputs**.

### Layer

The NLA reads block 41 of 62 where our workspace band is 46, and the layer sweep
shows selectivity varying materially across the band (the J-space-minus-random
difference flips sign between L24 and L57), so this is not negligible.
`configs/gemma3_27b_it_nla.yaml` pins `layer_band: [41, 41]` and
`export_nla_activations.py` refuses any other value.

### Site anchoring

Our reads are anchored to a sentence position; the NLA expects one vector per
row with no notion of which position it came from. The mapping is mechanical but
must be recorded, since "which token" is exactly the variable the site
dissociation turns on.

## Scoring

Whatever is chosen must be fixed before seeing outputs. This project has already
found five verdict labels that claimed more than their measurement supported,
and inventing a sixth by tuning a rubric after the fact would be worse than not
running the experiment.

**AR reconstruction MSE** is the pre-registerable number. The AR checkpoint
scores decode fidelity numerically — normalize predicted and gold vectors and
take MSE, equal to `2(1 - cos)`, where ~0.2 is good and ~2.0 is orthogonal — and
it supports the **minimal-pair** design directly: run the agent-role and
patient-role cells of the same family and ask whether their verbalizations
reconstruct to *different* activations. If the NLA is blind to role, the two
texts collapse to the same reconstruction. No rubric, no judge, and it inherits
the pairing the stimuli already have. Verbalizations are worth reporting
qualitatively alongside, but they are not the measurement.

String presence is too weak — both professions appear in both cells. An LLM
judge introduces a new instrument that would itself need validating.

**The controls have to come along**, matching how every other experiment here is
gated: the same minimal-pair contrast under a random activation and under a
non-participant entity's activation. Without them this reproduces exactly the
failure mode the primary analysis had to correct — a real-looking effect that
any strength-matched input would produce. The exported Parquet carries both role
cells of every family so all three contrasts are computable from it:

- **test** — within family, agent cell vs patient cell. Same words, same
  entities, only the role differs.
- **positive** — across families, entirely different sentences. If these do not
  separate either, the within-pair null is uninterpretable.
- **null** — shuffled pairing, cells from different families treated as a pair,
  giving the band a real within-pair difference must clear.
