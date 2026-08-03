# Results — Qwen3.6-27B, layer 48

**Read this before any file under `runs/`.**

`scripts/save_run.py` copies the whole of `data/results/` each time, so every
run directory carries forward the results of every earlier experiment. A given
JSON therefore appears in several directories, sometimes with a verdict string
that predates a fix. **This file names the authoritative artifact for each
experiment and states the corrected verdict.** Where a verdict differs from a
saved copy, the number was never wrong — only the label was, and the fix commit
is cited.

Model `Qwen/Qwen3.6-27B` · lens `neuronpedia/jacobian-lens`
(`qwen3.6-27b/jlens/Salesforce-wikitext`) · read layer 48 · 3× A100-40GB ·
bf16 · `device_map="auto"`

---

## Authoritative artifacts

| experiment | authoritative copy | verdict |
|---|---|---|
| Replication gate | `runs/gate/results/replication_gate.json` | **PASS** |
| RQ1 (decodability) | `runs/rq1-first/results/rq1_probe.json` | both sites, see below |
| RQ1 layer sweep | `runs/rq1-L{24,36,59}/results_L*/rq1_probe.json` | null holds 24–59 |
| RQ1 seed replication | `runs/rq1-seed42/results_S42/rq1_probe.json` | control reproduces |
| RQ1 6 pairs | `runs/rq1-6pair/results_6pair/rq1_probe.json` | replicates `entity_token` |
| E4 (recruitment) | `runs/e4-corrected/results/e4_recruitment.json` | `anti_transfer_both` |
| Within-pair `final_token` | `runs/within-pair-allpairs/results/within_pair.json` | `present_not_filler_general` |
| Within-pair `entity_token` | `runs/within-pair-entity/results/within_pair.json` | `present_not_filler_general` |
| RQ2 `final_token` | `runs/rq2/results/rq2_ablation.json` | not causally involved |
| RQ2 `entity_token` | `runs/rq2-entity-token/…` (+ patch below) | **not causally involved** |
| fit_directions | `runs/fit-directions/results/…` | site dissociation |
| calibrate | `runs/calibrate-exit3/…` | **exit 3** |
| Primary (E3) | `runs/logs/primary.log` | `significant_but_tiny` |
| Nonlinear probe | `runs/nonlinear-probe/results/nonlinear_probe.json` | `null_survives_nonlinearity` |

### Stale labels to ignore

- `runs/e4-first/`, and every directory saved before it — `e4_recruitment.json`
  reads `always_on`. Corrected to `anti_transfer_both` (commit `483ca1f`).
  0.281/0.263 are both **below** chance; the old rule never checked which side
  of 0.5 its inputs were on.
- `runs/nonlinear-probe/results/rq2_ablation.json` — reads
  `workspace_causally_involved: true` for `entity_token`. Corrected to **false**
  (commit `dd6e4fa`). See RQ2 below.
- Any `nonlinear_probe.json` reading `multiplicative_code_in_jspace` —
  corrected to `null_survives_nonlinearity` (commit `bb6172b`).

**Primary's verdict JSON exists only in `runs/logs/primary.log`.**
`run_primary.py` prints its analysis to stdout and never writes a summary file;
`runs/primary/` holds the raw `trials.jsonl.gz` only.

---

## 1. Replication gate — PASS

Directed modulation ("Think about X. Do Y"), J-lens score of each concept's own
token:

| prompt \ token | spider | piano | volcano |
|---|---|---|---|
| spider | **3.86** | 0.09 | 2.05 |
| piano | 0.85 | **5.88** | 2.27 |
| volcano | 0.68 | 1.57 | **4.84** |

Perfect diagonal, 3/3. Non-degeneracy: 16/16 atoms, ‖J‖/‖h‖ 0.21–0.24.

**Consequence: "the lens isn't reading" is eliminated for every null below.**

Incidental: the J-space atoms for the stimulus sentences are *meta-linguistic* —
这句话 "this sentence", 是谁 "who is", 动词 "verb", 反向 "reverse", ambiguous,
causal — not content words. Independently reproduces the interpretive-meta-token
phenomenon Nanda et al. reported on this model, with a different lens artifact.

## 2. Site dissociation — three methods agree

| method | `final_token` | `entity_token` |
|---|---|---|
| RQ1 decodability | below chance, all folds invert | above chance |
| Direction stability | **0.580–0.679, all six FAIL** the 0.8 threshold | **0.845–0.889, all pass** |
| Raw direction norms | 2.16–2.83 | 5.72–7.32 |

The role signal is at the **entity token**, not the final token. Established
before direction fitting ran, then confirmed by it.

## 3. RQ1 — decodability (leave-one-pair-out, 3 folds)

`final_token` — jspace 0.320 · orthogonal 0.281 · residual 0.283 ·
random_subspace 0.427. Every source **below chance**, all folds inverting.

`entity_token` — jspace 0.524 · orthogonal 0.626 · residual 0.594 ·
**random_subspace 0.713**.

A rank-matched random subspace beats J-space. See the anomaly in §8.

### Layer sweep — the null is not layer 48 alone

`entity_token`, same stimuli, read layer varied across the raw workspace band
(24–59). `runs/rq1-L24/`, `runs/rq1-L36/`, `runs/rq1-L59/`:

| layer | jspace | random_subspace | residual |
|---|---|---|---|
| 24 | 0.495 | 0.601 | 0.618 |
| 36 | 0.567 | 0.638 | 0.609 |
| **48** | **0.524** | **0.713** | **0.594** |
| 59 | 0.527 | 0.645 | 0.635 |

**J-space sits below the rank-matched random control at every layer in the
band.** The claim is "across the workspace band", not "at the layer we picked".

### Seed replication

`runs/rq1-seed42/`, seed 42 vs seed 0, `entity_token`:

| source | seed 0 | seed 42 |
|---|---|---|
| jspace | 0.524 | 0.524 |
| orthogonal | 0.626 | 0.626 |
| residual | 0.594 | 0.594 |
| random_subspace | 0.713 | 0.663 |

**Read this narrowly.** jspace, orthogonal and residual are *deterministic*
given the model — no randomness enters them, so their identity across seeds is
arithmetic, not evidence. The only quantity the seed moves is
`random_subspace` (the basis draw) and the control-task labels.

What it does buy: the capacity control still beats J-space (0.663 vs 0.524)
under an independent draw, so "J-space loses to a random subspace" is not one
lucky basis. It does **not** address run-to-run variance in the main sources,
because there is none to address.

## 4. Within-pair — the interpretive key

All 3 pairs, `final_token`, 300 families, splits held out **by family**:

| source | doctor→nurse | driver→passenger | teacher→student | **mean** |
|---|---|---|---|---|
| residual | 0.984 | 0.990 | 0.994 | **0.989** |
| orthogonal | 0.982 | 0.986 | 0.984 | **0.984** |
| random_subspace | 0.686 | 0.632 | 0.610 | 0.643 |
| **jspace** | 0.572 | 0.532 | 0.656 | **0.587** |

**Role is strongly linearly decodable from the residual stream and not from
J-space**, which sits at or below a same-rank random control.

Also run at `entity_token` (`runs/within-pair-entity/results/within_pair.json`,
300 families, same family-held-out splits):

| source | doctor→nurse | driver→passenger | teacher→student | **mean** |
|---|---|---|---|---|
| random_subspace | 0.836 | 0.860 | 0.856 | **0.851** |
| orthogonal | 0.788 | 0.860 | 0.796 | **0.815** |
| residual | 0.806 | 0.804 | 0.806 | **0.805** |
| **jspace** | 0.650 | 0.660 | 0.672 | **0.661** |

Same verdict at both sites — `present_not_filler_general`, with J-space below
the rank-matched control. Absolute accuracies are lower here than at
`final_token` (0.81 vs 0.99 for residual) but the ordering is unchanged, and
J-space is the only source failing to clear 0.8.

## 5. E4 — recruitment

jspace: role question 0.359, bag question 0.385, delta −0.026. No recruitment;
both conditions below chance. Verdict `anti_transfer_both`.

## 6. RQ2 — ablation

### `final_token`

```
binding_specific_deficit  +0.00024   CI [−0.0062, +0.0070]  spans zero
edit_magnitude             0.215 mean relative norm change, 7200 edits, landed
workspace_causally_involved  false
```

### `entity_token` — reads as positive, is not

```
binding_deficit           −0.0321    <- NEGATIVE
recall_deficit            −0.0464    <- NEGATIVE, larger
binding_specific_deficit  +0.0142    CI [0.0063, 0.0222], clears zero
```

**Ablation improved both tasks.** The positive difference is recall improving
*more* than binding improved — not binding being damaged. `binding − recall` is
a selectivity measure that presupposes damage; with both terms negative it
carries no causal claim. Corrected flag: **false**.

Also: the random control is matched on **rank** (`ablate_k` directions), not on
perturbation magnitude, and the two differ by **5.4×** here (0.234 vs 0.043)
because J-space ablation removes the most *active* directions. A J-space effect
exceeding the control is therefore not by itself evidence of localisation.

`difficulty_matched: false` at both sites (role 0.79 vs concept 0.59, gap 0.195).

## 7. Intervention — calibrate (exit 3) and E3

Push coefficient sweep at `entity_token`, target 0.5 log-odds shift:

```
coefficient  0.5     1.0     2.0     4.0     8.0
shift      −0.005  +0.008  +0.011  +0.005  +0.017
```

30–100× too small **and non-monotonic**: a 16× coefficient range produces the
same nothing. This is not the proposal's "too weak" case — pushing along fitted
J-space role directions has **no effect at any strength**.

E3 was then run at `push_coefficient=8.0` (largest grid value):
`significant_but_tiny`, pooled d **−0.096** against a pre-set meaningfulness
threshold of 0.5, `meaningful_constructions: []`, controls all ≈ −0.008.

**E3 must not be presented as a standalone causal null** — it ran under a
documented intervention-strength failure and its own proposal calls that case
uninterpretable. Its usable content is the **addressability dissociation**:
`strength_check_passes: true`, so IDENTITY_SWAP (a *concept-level* J-space edit)
demonstrably moves behaviour while role-direction pushes move nothing. Same
subspace, same layer, same machinery.

## 8. Nonlinear probe — the multiplicative-code objection

`entity_token`, 200 families, leave-one-pair-out:

| source | linear | quad | rff |
|---|---|---|---|
| jspace | 0.514 | **0.522** | 0.512 |
| orthogonal | 0.561 | 0.624 | 0.592 |
| residual | 0.545 | 0.605 | 0.596 |
| random_subspace | **0.719** | 0.620 | 0.521 |

`quad` includes random pairwise products `xᵢ·xⱼ`, so it **can** read a
tensor-product code. J-space peaks at **0.522** against chance 0.500.

**The multiplicative-encoding escape hatch is closed empirically**, in the
direction that supports the headline.

### Open anomaly

`random_subspace` scores **0.719** linear — above `residual` (0.545), the space
it is a 16-dimensional projection *of*. Reproduces RQ1's 0.713, and reproduces
again at seed 42 (0.663), so it is not one unlucky basis draw. Most likely a
fixed ridge penalty (`l2=1e-2`) across sources of wildly different
dimensionality: the low-dimensional projection is effectively better
regularised while the full 5120-dim residual overfits. **Cross-source
comparisons are therefore partly confounded by effective regularisation.** It
does not threaten the J-space null (J-space is at chance under every probe
family, which no regularisation story explains), but it needs a limitations
sentence.

**Not yet run:** `scripts/check_ridge_penalty.py` sweeps the penalty per source
over four orders of magnitude from a single GPU pass, and decides both
questions — whether J-space clears chance under *any* penalty (the
load-bearing check) and whether `residual` overtakes `random_subspace` once
each source is tuned (the anomaly). Until it runs, the regularisation account
above is a hypothesis, not a finding.

---

## The finding

Three independent lines agree, with the lens verified and capacity controls
throughout:

1. **Decoding** — role is strongly readable from the residual stream (0.989 at
   `final_token`, 0.805 at `entity_token`) and not from J-space (0.587 / 0.661,
   at or below a rank-matched random control at both sites, and at every layer
   in the 24–59 band).
2. **Transfer** — the role axis does not generalise across concept pairs;
   cross-pair transfer is systematically *inverted*, not merely at chance.
3. **Causal** — ablating J-space produces no binding deficit at either site,
   and pushing fitted J-space role directions moves the readout by ~0.01
   log-odds non-monotonically across a 16× coefficient range.

**Role-filler information is present and linearly readable in the residual
stream, but it is not carried by the J-space workspace and it is not
filler-general. Concept identity in J-space *is* addressable (IDENTITY_SWAP
propagates); role is not.**

## Limitations

1. **n = 3 concept pairs** → 3 folds for the primary table. `runs/rq1-6pair/`
   raises this to 6 on the real backend and replicates `entity_token` closely
   (jspace 0.534, random_subspace 0.723, residual 0.598); the other
   experiments still rest on 3.
2. **One model.** Layers are covered (24/36/48/59, §3) and the seed is
   replicated for what it can cover (§3), but every number in this file comes
   from Qwen3.6-27B. This is the largest remaining exposure.
3. **Three of four constructions are unvalidated drafts** — only
   `active_passive` is verified; cleft, relative-clause and dative are marked
   "pending team validation" in `templates.py`, yet all four appear in the
   primary table.
4. **E3 ran under an intervention-strength failure** (§7).
5. **Rank-matched, not norm-matched** ablation control (§6).
6. **Cross-source regularisation confound** (§8).
7. **Decodability ≠ use** everywhere except RQ2.
8. `difficulty_matched: false` in RQ2 at both sites.

## Note on verdict labels

Three verdict-labelling bugs were found and fixed during analysis: E4's
`always_on` (commit `483ca1f`), RQ2's `workspace_causally_involved`
(`dd6e4fa`), and the nonlinear probe's `multiplicative_code_in_jspace`
(`bb6172b`). All three shared one failure: a threshold comparing two quantities
without first asking whether either was distinguishable from chance or from
zero. **No measurement changed in any case — only which measurements were
allowed to be called a result.** Each fix carries a regression test pinning the
observed numbers.
