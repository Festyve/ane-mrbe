# Architecture

How the pipeline fits together, the binding-score math, and the contract each
module holds up. [RESULTS.md](../RESULTS.md) has the numbers; [RUNBOOK.md](../RUNBOOK.md)
has the GPU procedure.

The primary causal test is the **role-direction push**: a byte-identical push
along an in-house-fitted role axis `r_entity`, run in **both signs** as separate
uniform conditions. The lexical identity swap (doctor→nurse) is retained only as
the intervention-strength control, paired with the neutral probe. The readout is
**log-odds**, so a role-blind uniform push cancels exactly in the agent−patient
contrast and floor/ceiling effects cannot fake an interaction.

## Data flow

```
stimuli/generate.py ──────> data/stimuli/stimuli.jsonl          (ItemFamily records)
stimuli/fitting_corpus.py > data/stimuli/fitting_corpus.jsonl   (disjoint templates)
                                  │
scripts/fit_directions.py ───────┤  model.fitting_activation per (entity, role, site)
directions/fit.py ───────────────┤  difference-of-means -> r_entity (+ shuffled, LOO)
                                  ▼
                            data/directions/directions_{site}.npz  (+ stability JSON)
                                  │
scripts/calibrate.py ────────────┤  push_coefficient (fitting corpus), alpha (neutral probe)
                                  ▼
experiments/primary.py ──────────┤  for each family x cell x (edit x sign) x probe x site:
                                  │      model.answer_distribution(...)
                                  ▼
                            data/results/trials.jsonl            (TrialResult records)
                                  │
analysis/binding_score.py ───────┤  logit -> position-average -> both-sign crossover DiD
analysis/stats.py ───────────────┤  bootstrap CI / permutation p / d (+ d CI) / Holm
analysis/plots.py ───────────────┘  forest plot + per-condition probability plot

Secondary analyses, each with its own script + figure + JSON:
scripts/run_rq1.py ─> experiments/rq1_probe.py    (role decodability)
scripts/run_rq2.py ─> experiments/rq2_ablation.py (ablation deltas, causal use)
scripts/run_e4.py  ─> experiments/recruitment.py  (on-demand recruitment)
```

The runners are written against the `WorkspaceModel` protocol, so the whole
pipeline runs end-to-end on the GPU-free `DummyModel`. Its two ground-truth
modes are what validate the analysis before real weights are loaded: in
`binding` mode the pipeline must recover a ~2.2-logit binding score clearing the
null band with both per-sign gap changes negative, and in `bag` mode it must
recover ~0 inside the band. The dummy also plants a role direction (binding) or
pure noise (bag) in its fitting activations, so the direction-fitting pipeline
and its bootstrap stability check are validated the same way — in `bag` mode the
stability warning fires by design.

## The binding score

With `L(role, edit, sign)` the position-averaged logit P(entity) at the ROLE
probe:

```
dG(s) = [L(agent, push_s) - L(patient, push_s)] - [L(agent, no_edit) - L(patient, no_edit)]
BS_i  = -(dG(toward_agent) + dG(toward_patient)) / 2
```

A bag workspace adds the same log-odds increment to both role conditions, so
`dG(s) ≈ 0` and `BS ≈ 0`. A binding workspace shrinks the gap from opposite
sides under the two signs (the crossover), so `BS > 0`. A within-family
agent/patient label swap negates `BS_i` exactly, which makes the sign-flip
permutation test exact. The control pushes — non-participant, random,
shuffled-label — run the identical statistic and form the null band.

## Modules

All shared types live in `types.py` (stdlib only). All analysis is pure numpy
with no model dependency, so it is unit-testable on any machine.

### `types.py`
- `ConceptPair(entity, counterpart)` — `entity` is pushed and read;
  `counterpart` exists only for the IDENTITY_SWAP control and never appears in a
  sentence.
- `EditType` — `ROLE_PUSH` (primary), `NO_EDIT`, `NULL_NON_PARTICIPANT`,
  `RANDOM_DIRECTION`, `SHUFFLED_LABEL_DIRECTION`, `IDENTITY_SWAP`, and the two
  RQ2 ablations. `DIRECTION_PUSH_EDIT_TYPES` are the four that take a `PushSign`.
- `PushSign` — `TOWARD_AGENT` / `TOWARD_PATIENT`, separate uniform conditions,
  never conditioned on the sentence's own role label.
- `ProbeKind` — `ROLE`, `NEUTRAL`, `RECIPIENT` (dative only), `CONCEPT` (RQ2's
  recall control).

### `stimuli/`
- `templates.py` — the primary 2x2 quadruples per construction. Probes are
  byte-identical across cells.
- `vocab.py` — profession entities, CONCEPT-probe cues, and the
  non-participant candidates.
- `fitting_corpus.py` — 12 mirrored frames, template- and verb-disjoint from
  the primary set, role x position balanced. Tests assert the disjointness.
- `eval_corpus.py` — the held-out validation tier, disjoint from both others by
  sentence, frame ID, and normalized template.
- `qc.py` — the disjointness checks, exposed as `scripts/check_corpus.py`.

### `directions/fit.py`
`fit_role_direction` (unit difference-of-means), `fit_gradient_direction` (the
LRE-style steering estimator), `bootstrap_stability`, `shuffled_label_direction`
(overfitting control), `generic_loo_direction` (filler-general variant), and
`fit_all` / `save_directions` / `load_directions`.

### `interventions/edits.py`
`plan_edit(...) -> EditSpec`. Its module docstring is the canonical description
of both interventions: `h += s * c * unit(r)` for pushes, and the Gurnee et al.
coordinate swap for the identity control.

### `model/`
- `base.py` — the `WorkspaceModel`, `FittingActivationSource`,
  `ProbeActivationSource`, and `RecruitmentActivationSource` protocols.
- `dummy.py` — the synthetic backend. Its class docstring states every planted
  logit; the tests pin them.
- `qwen_jlens.py` — the real backend. Encodes the lens facts: the artifact is a
  per-layer averaged Jacobian `J_l`, the J-lens vector for token t is
  `J_l^T W_U[t]`, the J-space component comes from sparse pursuit, the identity
  swap patches in lens coordinates, and ABLATE_JSPACE zeroes the top-`ablate_k`
  active vectors. `layer_band` is an inclusive pair of RAW layer indices: edits
  apply across the band, reads use its top.

### `experiments/`
- `primary.py` — the sweep, plus `analyze()`: per-construction stats
  Holm-corrected within the family, a bootstrap CI on Cohen's d itself, the
  per-sign gap-change breakdown, both strength checks, and a `verdict` block.
- `calibrate.py` — `calibrate_push_coefficient` and `calibrate_identity_alpha`,
  both "smallest grid value that works", both on data disjoint from the primary
  stimuli.
- `rq1_probe.py` / `rq2_ablation.py` / `recruitment.py` — the three secondary
  experiments.
- `provenance.py` — the block stamped into every summary. Records what the RUN
  did, not what the config requested, and flags a dirty tree.

### `analysis/`
`binding_score.py` (above), `probes.py` (ridge probes + Hewitt–Liang control
task), `stats.py` (bootstrap CI, sign-flip permutation, Cohen's d + CI, Holm,
null band), `direction_sanity.py`, and `plots.py`.

### `scripts/`
Every entry point shares `model/factory.py` for backend construction and
`preflight_or_exit`, so backend readiness is checked BEFORE any sweep: **exit 2
always means "not configured or not fitted yet"**, and a mid-sweep error is a
genuine failure rather than a mislabeled setup problem.
`experiments.primary.validate_config` additionally rejects unusable sweep
configs before any model call.

## Design notes

Three deviations from the original proposal, all deliberate:

- The non-participant control pushes an absent **profession's** fitted
  direction rather than the proposal's "tuesday" — a weekday cannot bear a
  thematic role, so no role direction can be fitted for it.
- The dative's ROLE probe queries the giver, to keep scoring uniform. It also
  carries a second `RECIPIENT` readout so the dative is scored from both sides;
  its polarity is inverted, and `binding_score.PROBE_ORIENTATION` negates it so
  the two tables stay comparable. The dative is reported separately from the
  three agent/patient constructions either way.
- The source paper applies its identity swap at all token positions; we anchor
  at the injection site.

## CI

Every PR runs ruff, the unit tests, the full dry-run pipeline in both dummy
modes, and a check that the reviewable stimuli CSV has not drifted from the
templates. The ground-truth recovery check is a merge gate, not a local
courtesy.
