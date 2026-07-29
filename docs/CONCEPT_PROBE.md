# The CONCEPT probe: replacing RQ2's recall control

**Status: validated on Qwen2.5-1.5B against the committed stimuli.** Not yet
run on `Qwen/Qwen3.6-27B` — the numbers below establish that the probe is
*usable*, not that they transfer to the target model. Re-run
`scripts/check_concept_probe.py` on the GPU box before trusting any of it for
a real result.

## Why the old control had to go

RQ2 asks whether ablating J-space breaks *binding* or just breaks *the model*.
The recall control is what separates those: a role-blind task on the same
sentences that would also degrade if the damage were generic.

The NEUTRAL probe could not do that job.

```
The doctor treated the lawyer.
Question: Which professions are mentioned? Answer: The ___

correct iff  min(P(doctor), P(lawyer)) > P(nurse)
```

`doctor` and `lawyer` are in the text. `nurse` never appears. The task is
therefore "has this word occurred", not "what does this sentence say" — and no
ablation gentle enough to be informative about binding gets anywhere near
flipping it. The control read **exactly 1.0 under every condition**, so
`recall_deficit` was identically zero and

    binding_specific_deficit = binding_deficit - 0 = binding_deficit

The subtraction was a no-op. **No run using NEUTRAL could support a
"binding-specific" claim** — the headline claim of the proposal.

The tell that this was a design fault rather than a Qwen artifact: it
reproduced in `DummyModel`, whose probabilities are planted constants. A task
whose difficulty is identical no matter what answers it is not testing the
model.

## The replacement

```
The doctor treated the lawyer.
Question: Which one works in medicine?        Answer: The ___   -> doctor
Question: Which one argues cases in court?    Answer: The ___   -> lawyer

score = mean over both askings of  logit(P(correct)) - logit(P(incorrect))
```

Both candidate answers are **in the sentence**, so lexical presence cannot
answer it — the model has to know what the profession *is*. Cues share no stem
with the profession they identify (`PROFESSION_CUE` in `stimuli/vocab.py`), so
surface matching is closed off too.

**Role-blind by construction.** "The doctor treated the lawyer" and "The lawyer
treated the doctor" both answer *doctor*. The answer cannot move with role, so
the control cannot absorb part of the binding effect.

**Counterbalanced.** Asked once per participant, averaged. This is load-bearing:
a one-sided semantic probe is confounded by base rate and by primacy, and each
asking carries both in the opposite direction.

## Measurements

Qwen2.5-1.5B, `active_passive` families from `data/stimuli/primary_stimuli.csv`,
scored exactly as `rq2_ablation` scores them.

All three sections below are one run of:

```bash
python scripts/check_concept_probe.py --limit 80
```

### The control now has room to fall

| measure | mean | median | stdev |
|---|---|---|---|
| NEUTRAL margin (old) | +3.773 | +3.754 | 0.542 |
| CONCEPT margin (new) | +1.418 | +1.170 | 0.857 |

Same units, so this is a direct comparison. CONCEPT margin is positive on
**80/80** families — the model can do the task — at roughly a third of
NEUTRAL's margin, which is the headroom the control needs.

### Role-blindness is measured, not assumed

Signed agent-role margin minus patient-role margin, over all four
role x position cells:

```
mean signed shift : +0.0535
std error of mean :  0.0348
mean / SEM        :  1.54      -> indistinguishable from zero
systematic share  :  3.8% of the +1.418 margin
```

What movement exists is per-cell noise, and `rq2_ablation` averages the four
cells, which removes it. The probe is not tracking role.

### Known limitation: cue strength varies by pair

`strict acc` = all four cells **and** both askings correct — a deliberately
harsh bar (8 reads, all correct).

| pair | n | margin | strict acc |
|---|---|---|---|
| doctor/student | 10 | +0.610 | 0% |
| doctor/driver | 10 | +0.683 | 0% |
| doctor/lawyer | 10 | +0.695 | 0% |
| doctor/passenger | 10 | +0.973 | 40% |
| doctor/teacher | 10 | +1.435 | 50% |
| teacher/doctor | 10 | +1.435 | 50% |
| teacher/nurse | 10 | +2.625 | 100% |
| teacher/judge | 10 | +2.888 | 100% |

Every pair clears the `_MIN_BASELINE_MARGIN = 0.2` usability floor and every
pair is positive on average, so nothing here is unusable. But the spread is
4.7x, and the `doctor` cues are consistently the weak end. Weaker pairs have
proportionally less room to fall, so a real deficit will be harder to resolve
there. **Worth revisiting `PROFESSION_CUE["doctor"]` before the real run.**

## The scale bug this surfaced

Switching the control changed the units of the subtraction, and exposed a fault
that had been masked:

Role margins run several log-odds (baseline ~4.0); concept margins run around
1.4 by design. Under `bag` ground truth **both tasks collapse completely** — yet
subtracting raw log-odds leaves `4.04 - 1.37 = +2.67`, which reads as a large
binding-specific deficit and makes the run report causal involvement in the
one condition where there is none.

Deficits are therefore expressed as a **share of each task's own no-edit
baseline**, where 1.0 means total collapse:

```
binding_deficit = (base_role   - ablated_role)   / mean(base_role)
recall_deficit  = (base_recall - ablated_recall) / mean(base_recall)
```

Ground truth recovered:

| mode | binding lost | recall lost | specific | causally involved |
|---|---|---|---|---|
| binding | 0.999 | 0.563 | **+0.436** | `True` |
| bag | 0.999 | 1.003 | **−0.004** | `False` |

The old pass/fail version avoided this only by accident: accuracy is bounded,
so both tasks happened to share a 0–1 scale.

## What is still open

- **Not run on Qwen3.6-27B.** Everything above is 1.5B.
- **No ablation validation.** The probe has *headroom*; that it degrades under
  real J-space ablation by the right amount is untested, because that needs the
  GPU. First hour on the box.
- **`doctor` cues are weak** (see the table). Cheap to fix, worth doing first.
- **NEUTRAL is retained** and still used by `primary.py` for the IDENTITY_SWAP
  strength check, which genuinely wants a present-vs-absent contrast. Only
  RQ2's recall control moved.
