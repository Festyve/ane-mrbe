# Architecture

MVP skeleton for **testing role-filler binding in the J-space global workspace**
(Qwen3.6-27B + pre-fitted Jacobian lens).

## Data flow

```
stimuli/generate.py ──> data/stimuli/stimuli.jsonl        (ItemFamily records)
                              │
experiments/primary.py ───────┤  for each family x cell x edit x probe x site:
                              │      model.answer_distribution(...)
                              ▼
                        data/results/trials.jsonl          (TrialResult records)
                              │
analysis/binding_score.py ────┤  position-average -> difference-in-differences
analysis/stats.py ────────────┤  bootstrap CI / permutation p / Cohen's d / Holm
analysis/plots.py ────────────┘  forest plot + per-condition probability plot
```

The runner is written against the `WorkspaceModel` protocol, so the whole
pipeline runs end-to-end on the GPU-free `DummyModel`
(`scripts/run_primary.py --dry-run`). The dummy has two ground-truth modes —
`binding` and `bag` — which lets us **validate the analysis code against known
answers** before the real model is ever loaded: in `binding` mode the pipeline
must recover a large binding score that clears the null band; in `bag` mode it
must recover ~0 inside the band.

## Module contract (public API)

All shared types live in `types.py` (stdlib-only). All analysis is pure
numpy — no model dependencies — so it is unit-testable on any machine.

### `stimuli/vocab.py`
- `PROFESSION_ENTITIES: tuple[str, ...]` — candidate single-token entities.
- `OTHER_ENTITY_BY_PAIR: dict[str, str]` — pair_id -> default non-target participant.
- `validate_single_token(words, tokenizer) -> dict[str, bool]` — checks each word
  is one token under the target tokenizer. Callable once the Qwen tokenizer is
  available; pure interface until then.

### `stimuli/templates.py`
- `build_family(pair: ConceptPair, construction: Construction, other_entity: str,
  verb_lemma: str, family_index: int) -> ItemFamily`
  - ACTIVE_PASSIVE is **fully implemented** and must reproduce the proposal's
    worked example exactly (doctor/lawyer/treated):
    - (AGENT, FIRST): "The doctor treated the lawyer."
    - (AGENT, SECOND): "The lawyer was treated by the doctor."
    - (PATIENT, FIRST): "The doctor was treated by the lawyer."
    - (PATIENT, SECOND): "The lawyer treated the doctor."
  - DATIVE / CLEFT / RELATIVE_CLAUSE: draft templates behind the same
    interface, clearly marked `TODO(Monday): validate with team`. All four
    constructions form clean 2x2s.
  - Probes are byte-identical across the four cells. Role probe (draft wording,
    `TODO(Monday)`, per-construction): "Question: Who {verb_past} someone?
    Answer: The" for the transitive constructions; DATIVE uses "Question: Who
    {verb_past} a letter to someone? Answer: The" to match the ditransitive
    frame. Neutral probe: "Question: Which professions are mentioned? Answer: The".
- `VERBS_BY_CONSTRUCTION: dict[Construction, tuple[str, ...]]` — verb lemmas.

### `stimuli/generate.py`
- `generate_families(config: Config) -> list[ItemFamily]` — full crossing:
  concept_pairs x constructions x items_per_cell (verb / other entity vary
  round-robin over fixed pools per family_index; fully deterministic,
  independent of config.experiment.seed, which stimuli never consume).
- `save_families(families, path)` / `load_families(path)` — JSONL round-trip.

### `interventions/edits.py`
- `plan_edit(family: ItemFamily, edit_type: EditType, non_participant: str,
  alpha: float | None, seed: int) -> EditSpec`
  - REAL: source=pair.source, target=pair.target
  - NULL_NON_PARTICIPANT: source=non_participant, target=None (inject/remove a
    concept absent from the sentence)
  - RANDOM_DIRECTION: no concepts; seeded for reproducibility
  - NO_EDIT: empty spec
- Module docstring documents the coordinate-swap math the real backend must
  implement (Gurnee et al.): form `V = [v_s v_t]`, read lens coordinates
  `c = V+ h` (pseudoinverse), set `h_patched = h + V(sigma(c) - c)` where sigma
  swaps the two entries (optionally scaled by alpha). Preserves all components
  orthogonal to span(V).

### `model/dummy.py`
- `DummyModel(mode: str = "binding", seed: int = 0)` implements `WorkspaceModel`.
- Synthetic probabilities encode the proposal's worked example, plus a small
  position bias (+0.03 when the target is FIRST) so position-averaging is doing
  real work, plus seeded Gaussian jitter (sigma=0.02, clipped to [0.001, 0.999]).
  - ROLE probe, `binding` mode: P(target)=0.60 if REAL edit and the swapped
    concept sits in the probed role (AGENT); 0.05 if REAL and other role;
    0.01 under NO_EDIT and all controls.
  - ROLE probe, `bag` mode: P(target)=0.30 under REAL regardless of role;
    0.01 otherwise.
  - NEUTRAL probe (both modes): REAL edit moves P(target) to 0.55 and
    P(source) to 0.05 — the edit demonstrably "works" role-independently.
  - P(other) and P(source) fill sensible remainders; probabilities need not
    sum to 1 (full-softmax reads, per the protocol).

### `model/qwen_jlens.py`
- `QwenJLensModel(config: ModelConfig)` — **stub**. Constructor validates
  config and records the open Monday decisions in `missing_decisions`; the
  first `answer_distribution()` call raises `NotImplementedError` listing them
  (layer band, alpha, direction extraction via the "Tell me about {concept}"
  recipe, mean-subtracted over a 100-concept baseline) — this is the path
  scripts/run_primary.py catches for its exit-2 message. Structure the class so
  filling it in is mechanical: `_load_model()`, `_load_lens()`,
  `_concept_direction(word)`, `_apply_edit_hooks(edit, site)`,
  `answer_distribution(...)`.

### `analysis/binding_score.py`  (pure numpy; the math heart)
- `position_average(probs: dict[tuple[Role, Position], float]) -> dict[Role, float]`
- `family_binding_score(trials: list[TrialResult], target_token: str) -> float`
  - Uses ROLE-probe, FINAL_TOKEN-site trials of one family.
  - BS_i = [p(agent, REAL) - p(patient, REAL)] - [p(agent, NO_EDIT) - p(patient, NO_EDIT)],
    each p position-averaged, probabilities read at `target_token`.
- `control_binding_score(trials, target_token, edit_type) -> float` — same DiD
  with REAL replaced by a control edit (NULL_NON_PARTICIPANT or
  RANDOM_DIRECTION); these scores form the **null band**.
- `collect_scores(all_trials: list[TrialResult], site: InjectionSite)
  -> ScoreTable` where `ScoreTable` is a small dataclass:
  `real: dict[group_key, list[float]]`, `null_band: list[float]`;
  `group_key = (construction.value, pair_id)`.

### `analysis/stats.py`  (pure numpy)
- `bootstrap_ci(scores, n_resamples=10_000, ci_level=0.95, seed=0) -> (lo, hi)`
  — percentile CI over family-level scores.
- `permutation_pvalue(scores, n_permutations=10_000, seed=0) -> float`
  — within-family agent/patient label swap == sign-flip of each family's BS
  (document this equivalence in the docstring); two-sided.
- `cohens_d(scores) -> float` — one-sample d vs 0, ddof=1.
- `holm_bonferroni(pvalues: dict[K, float], alpha=0.05) -> dict[K, bool]`.
- `null_band(scores, ci_level=0.95) -> (lo, hi)` — percentile band of control scores.

### `analysis/plots.py`
- `forest_plot(score_table, stats_by_group, null_band, out_path)` — one row per
  group (construction x pair), mean BS +/- CI, shaded null band. The paper figure.
- `per_condition_plot(all_trials, target_token, out_path)` — raw agent vs
  patient P(target) under each edit type.

### `experiments/primary.py`
- `run_primary(config: Config, model: WorkspaceModel, families: list[ItemFamily])
  -> list[TrialResult]` — the full condition sweep. ROLE probe for all edit
  types; NEUTRAL probe additionally for REAL (the intervention-strength
  control). Writes `data/results/trials.jsonl`.
- `analyze(config, trials) -> dict` — scores, CIs, p-values (Holm-corrected),
  d, null band; saves figures; returns a JSON-serializable summary.

### `scripts/`
- `generate_stimuli.py --config configs/default.yaml [--out PATH]`
- `run_primary.py --config configs/default.yaml [--dry-run] [--dummy-mode binding|bag]`
  — `--dry-run` forces the DummyModel; prints the summary; exits nonzero if
  backend is qwen_jlens (not implemented yet).

## Testing
- `tests/test_templates.py` — active/passive 2x2 matches the worked example
  verbatim; probes byte-identical across cells; generate/save/load round-trip.
- `tests/test_binding_score.py` — hand-computed DiD cases (worked example:
  binding => BS = 0.55, bag => BS = 0.0); position bias cancels under averaging.
- `tests/test_stats.py` — bootstrap CI covers a known mean; permutation p small
  for a clearly shifted sample, large for a zero-centered one; Holm behaves on
  a crafted p-value set; Cohen's d matches a hand computation.
- `tests/test_pipeline.py` — end-to-end dry runs on a small config:
  `binding` mode recovers mean BS > null band; `bag` mode lands inside it.

## What is pinned vs open

| Pinned (implement fully) | Open (stub with TODO(Monday)) |
|---|---|
| 2x2 design, active/passive templates | dative/cleft/relative templates (drafts) |
| Binding score DiD + null band | layer band, alpha for Qwen3.6-27B |
| Bootstrap / permutation / d / Holm | probe wording (drafts marked) |
| Forest + per-condition plots | concept-direction extraction details |
| DummyModel end-to-end validation | single-token vocab validation (needs tokenizer) |
