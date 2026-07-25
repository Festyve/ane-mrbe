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
- `qwen_jlens.py` — the real backend, written to the SOURCE PAPER'S actual
  methods (Gurnee et al. 2026 §2), lazy torch imports, NOT yet validated on
  real weights. Lens facts it encodes: the artifact is a per-layer averaged
  Jacobian `J_l` (d_model x d_model); the J-lens vector for token t is
  `J_l^T W_U[t]` (W_U from the model itself); the J-space is sparse
  nonnegative combinations of those vectors, and an activation's J-space
  component comes from sparse pursuit (`_jspace_component`, k =
  `model.jspace_k`); the identity swap patches in lens coordinates using
  J-lens vectors directly (their §2.5 — no auxiliary concept-vector forwards
  needed); ABLATE_JSPACE zeroes the top-`ablate_k` active J-lens vectors
  (their §3.5.2). The artifact's schema is now CONFIRMED against the release
  (nested `J` mapping in a `.pt`; see the runbook) and covered by
  `tests/test_lens_loading.py`; the remaining unknown is ordinary
  first-contact bugs on real weights. Layer semantics: edits applied across the inclusive
  raw `layer_band`, reads at its top; their workspace is reindexed layers
  ~38–92 of 100 with single-layer analyses mid-band (~L75 reindexed). Site
  anchors: sentence-final token / the target entity's last token (also for
  the non-participant push — the site is a sentence position, not a property
  of the direction). Comparability note: several of the paper's swap
  experiments apply edits at ALL token positions; our site-anchored single
  position is a deliberate design difference.

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
- `fit_directions.py [--corpus PATH] [--allow-contaminated] [--entities a,b]
  [--directions-dir PATH] [--fitting-corpus-out PATH] [--export-pt]` — fitting
  corpus -> activations -> directions + stability warnings. Partial entity fits
  automatically use tagged output paths so they cannot overwrite canonical
  full-fit directions/corpus files. A hand-written corpus colliding with the
  primary set is REFUSED unless --allow-contaminated (which stamps the output
  summary). --export-pt additionally writes per-entity `.pt` bundles with
  model/layer/corpus/commit provenance (directions/export_pt.py) for the
  torch-based handoff.
- `export_directions_pt.py [--in DIR] [--out DIR] [--fit-corpus PATH]
  [--entity a,b]` — convert already-fitted `directions_{site}.npz` to
  `{entity}_role_direction.pt` without re-running the model, recording the
  source corpus and repository commit in each payload.
- `direction_sanity.py --eval-corpus PATH [--fit-corpus PATH]
  [--directions-dir PATH] [--entity a,b]` — held-out separation (AUC / midpoint
  accuracy / Cohen's d + strip scatter) and pairwise direction cosines
  (analysis/direction_sanity.py). It refuses eval examples that reuse fitting
  sentences, frame IDs, or normalized surface templates; results in
  `docs/DIRECTION_SANITY.md`.
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

0. FIRST, on any laptop: `pip install '.[model]'` then
   `python scripts/smoke_test.py` — runs the ENTIRE real backend (loading,
   lens, pursuit, hooks, all edits, fitting, calibration, all three
   experiments) on a tiny open model with a synthetic identity-Jacobian lens
   (= the logit lens, per the paper's §2.4). Numbers are not science; PASS
   means the code paths work, so GPU time is spent on the experiment rather
   than on typos.
1. `pip install '.[model]'` (+ `bitsandbytes` if `load_in_4bit`).
2. Check `hf auth whoami`. An expired/invalid stored token makes the Hub
   return 401 *Repository Not Found* even for these public repos, which reads
   like a wrong model id — `hf auth login --force`, or clear the token to
   fetch anonymously.
3. The lens is `neuronpedia/jacobian-lens`, one lens per model under
   `{model}/jlens/{corpus}/`; `model.lens_subpath` scopes the download and the
   file search to ours (`qwen3.6-27b/jlens/Salesforce-wikitext`, 3.3 GB).
   Leave it set — the repo totals ~57 GB and an unscoped fetch pulls every
   other model's Jacobians. Verified schema: a torch `.pt` holding
   `{"J": {int_layer: (5120, 5120) fp16}, "source_layers": [0..62],
   "d_model": 5120, "n_prompts": 1000}`.
4. Pin `model.layer_band` in RAW layer indices. Guidance from the source
   paper: workspace = reindexed layers ~38–92 of 100; single-layer swap
   analyses sit mid-band (~L75 reindexed). Convert via
   raw = round(reindexed/100 * n_layers). For Qwen3.6-27B (**64 layers,
   d_model 5120**) that is raw **24–59**, mid-band **~48**. The lens covers
   raw layers **0–62**, so the whole band is available.
5. Validate the vocab: `stimuli.vocab.validate_single_token` on all entities.
   DONE for Qwen3.6-27B — all 12 PROFESSION_ENTITIES encode to a single
   token with a leading space.
   (Correction to an earlier version of this step: there is no "n=1000
   attested concepts" list to check the pairs against. The artifact's
   `n_prompts: 1000` is the number of prompts the per-layer Jacobian was
   AVERAGED OVER, not a concept whitelist — the J-lens dictionary is
   `J_l^T W_U[t]` over the whole vocabulary, so any single token has a
   J-lens vector.)
6. `scripts/fit_directions.py` — check stability per entity per site.
7. `scripts/calibrate.py` — copy alpha / push_coefficient into the config.
8. `scripts/run_primary.py` with `backend: qwen_jlens`; then `run_rq1.py`,
   `run_rq2.py`.

## Proposal ↔ repo map

| Proposal element | Where it lives |
|---|---|
| Role directions r_entity (diff-of-means over J-space projections, per site) | `directions/fit.py` + backend `fitting_activation` (sparse-pursuit component) |
| Fitting corpus, disjoint at template level; stability pilot | `stimuli/fitting_corpus.py`; bootstrap check in `directions/fit.py`; gate in `scripts/fit_directions.py` |
| 2x2 stimuli x 4 constructions, worked example verbatim | `stimuli/templates.py` (pinned by tests) |
| Both-sign push, byte-identical, never role-conditioned | `experiments/primary.py` sweep + `interventions/edits.py` |
| Binding score BS_i = -1/2[dG(+)+dG(-)] in log-odds | `analysis/binding_score.py` (proposal §6, as amended) |
| Controls: no-edit / neutral-probe swap / non-participant / random / shuffled | `types.EditType` + sweep; strength check verdict in `analyze()` |
| Bootstrap CI, permutation, Cohen's d + CI on d, Holm | `analysis/stats.py` |
| Outcome shapes (positive / tiny / clean-negative / underpowered) | `_verdict` in `experiments/primary.py` |
| RQ1 probe (3 sources, Hewitt–Liang selectivity, split by pair) | `analysis/probes.py`, `experiments/rq1_probe.py` |
| RQ2 ablation (J-space vs matched random subspace, matched recall task) | `experiments/rq2_ablation.py` + backend ablation edits |
| Forest / per-condition / selectivity / ablation-delta figures | `analysis/plots.py` |
| Calibration ("smallest value that works", fitting corpus only) | `experiments/calibrate.py` |
| Leave-one-out generic direction robustness | `directions/fit.py` (`generic_loo`), `directions.variant` config |

Deliberate deviations from the proposal text (team-flagged): the
non-participant control pushes an absent PROFESSION's fitted direction
(a weekday cannot bear a thematic role — proposal's "tuesday" example is
unfittable under the role-direction design); the dative's ROLE probe queries
the giver to keep scoring uniform, with dative reported separately either way.

The dative additionally carries a second readout, `ProbeKind.RECIPIENT`
("Who was handed the letter by someone?"), so the recipient — the participant
the ROLE probe never asks about — is scored too. It rides along with ROLE on
every push and on the no-edit baseline, giving the dative a full 12-cell score
and its own null band from the recipient side. Its polarity is inverted
(P(entity) rises as the entity becomes more PATIENT-like), so
`binding_score.PROBE_ORIENTATION` negates it and the two tables stay directly
comparable. Non-dative families carry `recipient_probe == ""` and the sweep
skips them. OPEN: which readout the dative's headline score uses — currently
both are computed and neither is privileged.

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
| 2x2 design, all four construction 2x2s (drafts flagged) | probe wording sign-off (incl. the new dative RECIPIENT probe, and which dative readout is headline) |
| Role-push edit family, both signs, all controls | layer_band for Qwen3.6-27B (raw index; candidate band 24–59, mid ~48) |
| Log-odds crossover DiD + null band + per-sign breakdown | alpha, push_coefficient (calibration) |
| Direction fitting + stability + shuffled + LOO | ~~lens artifact KEY NAMES~~ (confirmed + tested) |
| Fitting corpus (disjoint, balanced) | exemplars_per_role (stability pilot) |
| Bootstrap / permutation / d / Holm | ~~single-token vocab~~ (verified on Qwen3.6-27B); no attestation list exists |
| Dummy end-to-end validation, both modes | non-participant = absent profession (review) |
| Backend math per Gurnee et al. §2 (J_l, pursuit, swap, ablation) | first-contact validation on real weights |
