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
| Ridge penalty sweep | `runs/ridge-penalty/results/ridge_penalty_sweep.json` | `jspace_null_survives_penalty_sweep` |

### Stale labels to ignore

- `runs/e4-first/`, and every directory saved before it — `e4_recruitment.json`
  reads `always_on`. Corrected to `anti_transfer_both` (commit `483ca1f`).
  0.281/0.263 are both **below** chance; the old rule never checked which side
  of 0.5 its inputs were on.
- **RQ2 `workspace_causally_involved` — now corrected IN PLACE, not just noted.**
  Two commits changed this verdict rule: `dd6e4fa` (two improvements must not
  read as involvement) and `e61aa81` (require a magnitude of binding damage,
  `_MIN_BINDING_DEFICIT = 0.05`). Three archived files were written under the
  older rules and read `true`:
  `runs/nonlinear-probe`, `runs/rq2-entity-token`, `runs/gemma3-12b-pilot`.
  All three now read **false**, rewritten by `scripts/reanalyze_rq2.py`, each
  carrying a `verdict_rescored` block recording the previous value. **No delta
  was recomputed**: the measurements are exactly as originally run. Re-check the
  whole archive at any time with `python scripts/reanalyze_rq2.py --all`, which
  exits 1 if any verdict has drifted from the current rule.
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
uninterpretable.

### ⚠️ The addressability dissociation is RETRACTED

Earlier versions of this file claimed a dissociation here: `strength_check_passes:
true`, so IDENTITY_SWAP (a concept-level J-space edit) "demonstrably moves
behaviour" while role pushes move nothing. **The data does not support that.**

The observed strength check on Qwen:

```
P(counterpart)  no_edit 0.00269 -> swap 0.00393   (+0.00124)
P(entity)       no_edit 0.18507 -> swap 0.18091   (-0.00417)
```

The counterpart rises by **0.12 percentage points**, from 0.27% to 0.39%. The
entity does not meaningfully fall. **Nothing happened.** The check returned
`true` because it tested only the *signs* of the two shifts, with no magnitude
floor — while `calibrate_identity_alpha`, asking the same question on the same
model, reported an intervention-strength **failure** at its `min_prob_shift =
0.05`. The two disagreed and the looser one was believed.

Fixed: `_MIN_COUNTERPART_SHIFT = 0.05` in `experiments/primary.py`, matching
calibration, with a regression test pinning the numbers above. Under the
corrected check **Qwen's IDENTITY_SWAP does not pass**, so E3 has no
interpretable content at all — neither a causal null nor a dissociation.

Gemma-3-12B is the instructive contrast (`runs/gemma3-12b-lre/`): there the
entity drops hard (0.2122 → 0.0961) while the counterpart still barely moves
(0.00544 → 0.01021). The swap *damages* the entity readout without installing
the counterpart — consistent with the identity-overlap account, and not a
working swap either. Its calibration recorded the same `alpha_failure`.

**Consequence: the project currently has no positive causal result.** Every
J-space claim is a null. This is the single most important thing to know before
choosing a framing.

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

### The anomaly, resolved — §9

## 9. Ridge penalty sweep — the regularisation confound, settled

`entity_token`, 200 families, leave-one-pair-out at seven penalties
(`runs/ridge-penalty/results/ridge_penalty_sweep.json`):

| source | 1e-04 | 1e-03 | 1e-02 | 1e-01 | 1e+00 | 1e+01 | 1e+02 |
|---|---|---|---|---|---|---|---|
| jspace | 0.510 | 0.511 | **0.514** | 0.518 | 0.589 | 0.645 | 0.639 |
| orthogonal | 0.561 | 0.561 | **0.561** | 0.561 | 0.564 | 0.586 | 0.722 |
| residual | 0.545 | 0.545 | **0.545** | 0.545 | 0.546 | 0.565 | 0.677 |
| random_subspace | 0.724 | 0.721 | **0.719** | 0.715 | 0.714 | 0.687 | 0.678 |

(bold = the fixed `l2=1e-2` every other experiment used)

**Never rank sources on a max-over-penalties column** — it selects the penalty
on the same folds it reports. Two honest views:

| source | nested CV | matched `l2=1e+02` |
|---|---|---|
| orthogonal | 0.722 | 0.722 |
| random_subspace | 0.703 | 0.678 |
| residual | 0.677 | 0.677 |
| **jspace** | **0.640** | **0.639** |

**Two conclusions.**

1. **The J-space null survives.** J-space is *last* at all seven penalties, and
   last under nested CV. It never clears the rank-matched control, which is the
   comparison the localisation claim rests on.

2. **The §8 anomaly was regularisation.** At `l2=1e+02`, `residual` 0.677 and
   `random_subspace` 0.678 are level, and `orthogonal` 0.722 exceeds both. The
   "projection beats its source" impossibility appears only at weak penalties,
   where the 5120-dim sources overfit and the 16-dim one does not — and
   `l2=1e-2` sat in that regime. A 0.026 gap survives under nested CV (0.703 vs
   0.677), so "largely explained" rather than "fully explained".

**Consequence for wording.** J-space runs 0.514 → 0.645 across the grid, so
**"J-space is at chance" is too strong and should not appear in the paper.**
The claim the controls support is that **J-space carries less role information
than an arbitrary subspace of the same rank.**

---

## The finding

Three independent lines agree, with the lens verified and capacity controls
throughout:

1. **Decoding** — role is strongly readable from the residual stream (0.989 at
   `final_token`, 0.805 at `entity_token`) and less so from J-space (0.587 /
   0.661), which sits at or below a rank-matched random control at both sites,
   at every layer in the 24–59 band, and at every ridge penalty across four
   orders of magnitude (§9). **Not "at chance"** — under heavy regularisation
   J-space reaches 0.645 — but always *below the capacity control*.
2. **Transfer** — the role axis does not generalise across concept pairs;
   cross-pair transfer is systematically *inverted*, not merely at chance.
3. **Causal** — ablating J-space produces no binding deficit at either site,
   and pushing fitted J-space role directions moves the readout by ~0.01
   log-odds non-monotonically across a 16× coefficient range.

**Role-filler information is present and linearly readable in the residual
stream, but the J-space workspace is not a privileged locus for it — J-space
carries less role information than an arbitrary subspace of the same rank —
and it is not filler-general.**

~~Concept identity in J-space *is* addressable (IDENTITY_SWAP propagates); role
is not.~~ **Retracted — see §7.** The strength check that licensed this passed
on a 0.12-percentage-point counterpart shift and fails under the corrected
magnitude threshold.

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
6. **Every experiment except §9 ran at the fixed `l2=1e-2`**, which §9 shows is
   a weak-penalty regime where high-dimensional sources overfit. The J-space
   null is unaffected (it holds at every penalty), but the *absolute* accuracies
   quoted for `residual` and `orthogonal` elsewhere in this file understate them
   — 0.545 vs 0.677 for `residual` at `entity_token`. Re-running RQ1 under
   nested CV would make the whole table internally consistent.
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
