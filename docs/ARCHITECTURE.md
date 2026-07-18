# Architecture

Codebase for **testing role-filler binding in the J-space global workspace**
(Qwen3.6-27B + pre-fitted Jacobian lens).

Design pivot (matches the current proposal): the PRIMARY causal test is the
**role-direction push** — a byte-identical push along an in-house-fitted role
axis `r_entity`, run in **both signs** as separate uniform conditions. The
lexical identity swap (doctor→nurse) is retained **only** as the
intervention-strength control, paired with the neutral probe. The readout is
**log-odds** of the entity's role-probe answer, so a role-blind uniform push
cancels exactly in the agent−patient contrast and floor/ceiling effects cannot
fake an interaction.

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

Secondary analyses (proposal §4), each with its own script + figure + JSON:
scripts/run_rq1.py ─> experiments/rq1_probe.py    (role decodability; selectivity plot)
scripts/run_rq2.py ─> experiments/rq2_ablation.py (ablation deltas; causal-use check)
```

The runner is written against the `WorkspaceModel` protocol, so the whole
pipeline runs end-to-end on the GPU-free `DummyModel`
(`scripts/run_primary.py --dry-run`). The dummy has two ground-truth modes —
`binding` and `bag` — which lets us **validate the analysis code against known
answers** before the real model is ever loaded: in `binding` mode the pipeline
must recover a ~2.2-logit binding score that clears the null band with both
per-sign gap changes negative (the crossover); in `bag` mode it must recover
~0 inside the band. The dummy also implements `FittingActivationSource` with a
planted role direction (binding) or pure noise (bag), so the direction-fitting
pipeline and its bootstrap stability pilot check are validated the same way —
in `bag` mode the stability warning fires by design.

## The binding score (analysis/binding_score.py)

With `L(role, edit, sign)` = position-averaged logit P(entity) at the ROLE
probe:

```
dG(s) = [L(agent, push_s) - L(patient, push_s)] - [L(agent, no_edit) - L(patient, no_edit)]
BS_i  = -(dG(toward_agent) + dG(toward_patient)) / 2
```

Bag workspace ⇒ each push adds the same log-odds increment to both role
conditions ⇒ `dG(s) ≈ 0` ⇒ `BS ≈ 0`. Binding workspace ⇒ the gap shrinks from
opposite sides under the two signs (crossover) ⇒ `BS > 0`. A within-family
agent/patient label swap negates BS_i exactly, so the sign-flip permutation
test is exact. Control pushes (non-participant, random, shuffled-label) run
the identical statistic and form the null band.

## Module contract (public API)

All shared types live in `types.py` (stdlib-only). All analysis is pure
numpy — no model dependencies — so it is unit-testable on any machine.

### `types.py`
- `ConceptPair(entity, counterpart)`: `entity` is pushed/read; `counterpart`
  exists only for the IDENTITY_SWAP strength control and never appears in a
  sentence. `pair_id` = `"entity->counterpart"`.
- `EditType`: `ROLE_PUSH` (primary), `NO_EDIT`, `NULL_NON_PARTICIPANT`,
  `RANDOM_DIRECTION`, `SHUFFLED_LABEL_DIRECTION`, `IDENTITY_SWAP` (control
  only). `DIRECTION_PUSH_EDIT_TYPES` = the four that take a `PushSign`.
- `PushSign`: `TOWARD_AGENT` / `TOWARD_PATIENT` — separate uniform conditions,
  never conditioned on the sentence's own role label.
- `EditSpec`: entity/sign/coefficient for pushes; swap_source/swap_target/alpha
  for the identity swap; seed for the random direction.
- `TrialResult`: + `push_sign` (None for unsigned edits).

### `stimuli/`
- `templates.py` — the primary 2x2 quadruples per construction; ACTIVE_PASSIVE
  reproduces the proposal's worked example verbatim (tests pin it). Probes are
  byte-identical across cells.
- `vocab.py` — profession entities; `NON_PARTICIPANT_CANDIDATES` for the null
  push (deviation from the proposal's "tuesday": a weekday cannot bear a
  thematic role, so the absent concept is an absent PROFESSION with its own
  fitted direction — flagged for team review).
- `fitting_corpus.py` — 12 mirrored frames (template-level disjoint from the
  primary set; disjoint verb pool; role x position balanced), generating
  `exemplars_per_role` sentences per entity per role, each carrying its own
  role probe for calibration. Tests assert the disjointness.
- `generate.py` — deterministic primary-set generation + JSONL round-trip.

### `directions/fit.py` (pure numpy)
- `fit_role_direction(agent_acts, patient_acts)` — unit difference-of-means.
- `bootstrap_stability(...)` — mean cosine of resampled fits vs the full fit
  (the proposal's fitting-corpus pilot check).
- `shuffled_label_direction(...)` — the direction-overfitting control (note in
  the docstring: conservative under strong signal).
- `generic_loo_direction(...)` — leave-one-entity-out filler-general variant.
- `fit_all(...) -> FittedDirections`; `save_directions`/`load_directions`
  round-trip `data/directions/directions_{site}.npz` (+ JSON summary).

### `interventions/edits.py`
- `plan_edit(family, edit_type, sign, non_participant_candidates, alpha,
  coefficient, seed) -> EditSpec`. Module docstring is the canonical
  description of both maths: `h += s * c * (B @ r_unit)` for pushes (B = lens
  decoder), and the Gurnee et al. coordinate swap for the identity control.
- `choose_non_participant(family, candidates)` — first candidate absent from
  the family's sentences.

### `model/`
- `base.py` — `WorkspaceModel` (answer_distribution) +
  `FittingActivationSource` (fitting_activation) protocols.
- `dummy.py` — synthetic backend, logit-space tables (see class docstring for
  exact numbers), content-hash-seeded jitter, planted fitting activations.
- `qwen_jlens.py` — the real backend, fully shaped, lazy torch imports, NOT
  yet validated on real weights. Open unknown: the released lens's artifact
  schema (`_load_lens` documents the expected `layer_{L}.encoder/decoder`
  keys and fails loudly). Layer semantics: edits applied across the inclusive
  `layer_band`, reads at its top. Site anchors: sentence-final token / the
  target entity's last token (also for the non-participant push — the site is
  a sentence position, not a property of the direction). Identity directions
  via the "Tell me about {concept}" recipe, mean-subtracted over a
  100-concept baseline.

### `experiments/`
- `primary.py` — the sweep: pushes x both signs at the ROLE probe; NO_EDIT at
  ROLE + NEUTRAL; IDENTITY_SWAP at NEUTRAL only. analyze() adds
  `gap_change_by_sign` (descriptive crossover breakdown),
  `neutral_strength_check` (the load-bearing control: fails ⇒ any null is
  uninterpretable), per-`constructions` stats blocks (Holm-corrected within
  the construction family — the positive-result criterion quantifies over
  constructions), a bootstrap CI **on Cohen's d itself**, and a `verdict`
  block classifying the outcome per the proposal's Benchmarks/Ideal Results:
  positive_binding / significant_but_tiny / suggestive_not_conclusive /
  clean_negative / inconclusive_underpowered /
  uninterpretable_strength_failure.
- `calibrate.py` — `calibrate_push_coefficient` (fitting corpus only, never
  primary stimuli) and `calibrate_identity_alpha` (neutral probe), both
  "smallest grid value that works".
- `rq1_probe.py` — RQ1: leave-one-pair-out ridge probes over three activation
  sources (J-space component / orthogonal remainder / full residual) with
  Hewitt–Liang control-task selectivity (analysis/probes.py). Decodability,
  not use; a linear null is nearly uninformative (Smolensky caveat) — the
  summary carries that note.
- `rq2_ablation.py` — RQ2: NO_EDIT vs ABLATE_JSPACE vs
  ABLATE_RANDOM_SUBSPACE (matched-dimension capacity control) scored on the
  binding task (ROLE probe: does the higher-ranked participant match the
  true agent?) and the recall task (NEUTRAL probe: do both mentioned
  participants outrank the absent token?) over the SAME sentences —
  difficulty matching by construction. Causal involvement = the J-space
  binding-specific deficit exceeding both zero and the random bar.

### `analysis/`
- `binding_score.py` — see above. `collect_scores` groups by
  (construction, pair), pools the null band, records per-sign gaps.
- `stats.py` — bootstrap CI, sign-flip permutation, Cohen's d,
  Holm-Bonferroni, null band. Unchanged math; the unit is the family score.
- `plots.py` — forest plot (one row per construction x pair over the shaded
  null band) and the per-condition P(entity) plot where the crossover is
  visible directly.

### `scripts/`
All five entry points share `model/factory.py` (build_model + the common
flags + `preflight_or_exit`): backend readiness is checked BEFORE any sweep,
so exit 2 always means "not configured/fitted yet" and a mid-sweep error is a
genuine failure, never mislabeled. `experiments.primary.validate_config`
additionally rejects unusable sweep configs (single push sign, RQ2 ablation
edit types) before any model call.
- `generate_stimuli.py --config configs/default.yaml [--out PATH]`
- `fit_directions.py [--corpus PATH] [--allow-contaminated]` — fitting
  corpus -> activations -> directions + stability warnings. A hand-written
  corpus colliding with the primary set is REFUSED unless
  --allow-contaminated (which stamps the output summary).
- `calibrate.py` — writes data/calibration.json; copy the values into the
  config's model section. Exit 3 = intervention-strength failure.
- `run_primary.py` / `run_rq1.py` / `run_rq2.py` — the experiments.
- `check_corpus.py --corpus ... [--against ...] [--require-balance]` —
  corpus QC gate (imbalance is a warning unless --require-balance, so eval
  grids like 8/7/7/8 pass while fitting corpora can be held to balance).

### CI (.github/workflows/ci.yml)
Every PR runs ruff, the unit tests, and the full dry-run pipeline in BOTH
dummy modes — the ground-truth recovery check is a merge gate, not a local
courtesy.

## GPU-day runbook

1. `pip install '.[model]'` (+ `bitsandbytes` if `load_in_4bit`).
2. Verify the HF model id and download the lens; inspect its keys; adapt
   `_load_lens`'s key mapping if the release schema differs.
3. Pin `model.layer_band` (single layer `[L, L]` = source-paper comparability).
4. Validate the vocab: `stimuli.vocab.validate_single_token` on all entities.
5. `scripts/fit_directions.py` — check stability per entity per site.
6. `scripts/calibrate.py` — copy alpha / push_coefficient into the config.
7. `scripts/run_primary.py` with `backend: qwen_jlens`.

## Testing
- `tests/test_templates.py` — worked example verbatim; probes byte-identical
  and entity-free; generation deterministic; JSONL round-trip.
- `tests/test_fitting_corpus.py` — verb + template disjointness from the
  primary set; role x position balance; round-trip.
- `tests/test_directions.py` — planted-direction recovery; stability
  separates signal from noise; shuffled control loses role signal (statistical
  contract); LOO correctness; storage round-trip.
- `tests/test_edits.py` — plan_edit contracts (signs required for pushes,
  non-participant skips sentence entities, seeds recorded).
- `tests/test_binding_score.py` — hand-computed crossover (BS = 2.2) and bag
  (BS = 0) cases; the raw-probability phantom-effect demonstration; position
  bias cancellation; missing-cell errors; grouping.
- `tests/test_stats.py` — bootstrap/permutation/Holm/d + d-CI contracts.
- `tests/test_probes.py` — ridge probe separates separable data; leave-one-
  pair-out generalization; control task at chance; chance without signal.
- `tests/test_rq1_rq2.py` — RQ1 localizes role in the mode-appropriate
  subspace; RQ2 shows the binding-specific deficit only in binding mode.
- `tests/test_pipeline.py` — end-to-end dry runs: binding mode recovers the
  planted effect above the null band with both signed gaps negative and a
  d CI excluding zero; bag mode lands inside the band and classifies as
  clean_negative; strength check passes in both modes.

## What is pinned vs open

| Pinned (implemented + tested) | Open (team decisions / GPU day) |
|---|---|
| 2x2 design, all four construction 2x2s (drafts flagged) | probe wording sign-off |
| Role-push edit family, both signs, all controls | layer_band for Qwen3.6-27B |
| Log-odds crossover DiD + null band + per-sign breakdown | alpha, push_coefficient (calibration) |
| Direction fitting + stability + shuffled + LOO | lens artifact schema (adapt `_load_lens`) |
| Fitting corpus (disjoint, balanced) | exemplars_per_role (stability pilot) |
| Bootstrap / permutation / d / Holm | single-token vocab validation (needs tokenizer) |
| Dummy end-to-end validation, both modes | non-participant = absent profession (review) |
