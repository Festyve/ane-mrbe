"""End-to-end dry runs against DummyModel ground truth (docs/ARCHITECTURE.md, Testing).

binding mode must recover a large pooled binding score (the ~2.2-logit
crossover encoded in the dummy) that clears the control null band, with both
per-sign gap changes negative; bag mode must land near 0 inside the band.
Both modes must pass the neutral-probe strength check (the identity swap
"works" role-independently in either world). This validates the analysis code
against known answers before the real model is ever loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from jspace_binding.config import AnalysisConfig, Config, PathsConfig, StimuliConfig
from jspace_binding.experiments.primary import analyze, run_primary
from jspace_binding.model.dummy import DummyModel
from jspace_binding.stimuli.templates import VERBS_BY_CONSTRUCTION, build_family
from jspace_binding.stimuli.vocab import OTHER_ENTITY_BY_PAIR
from jspace_binding.types import (
    DIRECTION_PUSH_EDIT_TYPES,
    Construction,
    ItemFamily,
    PushSign,
)


def _tiny_config(tmp_path: Path) -> Config:
    return Config(
        stimuli=StimuliConfig(items_per_cell=2),
        analysis=AnalysisConfig(n_bootstrap=500, n_permutation=500),
        paths=PathsConfig(
            stimuli=tmp_path / "stimuli.jsonl",
            fitting_corpus=tmp_path / "fitting_corpus.jsonl",
            directions=tmp_path / "directions",
            calibration=tmp_path / "calibration.json",
            results=tmp_path / "results",
            figures=tmp_path / "figures",
        ),
    )


def _families(config: Config) -> list[ItemFamily]:
    """All four constructions, at the tiny config's scale."""
    return [
        build_family(
            pair=pair,
            construction=construction,
            other_entity=OTHER_ENTITY_BY_PAIR.get(pair.pair_id, "lawyer"),
            verb_lemma=VERBS_BY_CONSTRUCTION[construction][
                index % len(VERBS_BY_CONSTRUCTION[construction])
            ],
            family_index=index,
        )
        for construction in Construction
        for pair in config.stimuli.concept_pairs
        for index in range(config.stimuli.items_per_cell)
    ]


def _run(tmp_path: Path, mode: str) -> dict:
    config = _tiny_config(tmp_path)
    families = _families(config)
    assert families, "the pinned active_passive construction must always build"
    trials = run_primary(config, DummyModel(mode=mode, seed=0), families)
    # Per cell x site: each direction-push edit runs both signs at the ROLE
    # probe, NO_EDIT runs ROLE + NEUTRAL, IDENTITY_SWAP runs NEUTRAL only.
    n_push = sum(1 for e in config.experiment.edit_types if e in DIRECTION_PUSH_EDIT_TYPES)
    n_signs = len(config.experiment.push_signs)
    per_cell_site = n_push * n_signs + 2 + 1
    # Dative families carry a RECIPIENT probe too, which rides along with ROLE
    # on every push (both signs) and on the no-edit baseline.
    per_cell_site_recipient = n_push * n_signs + 1
    n_recipient_families = sum(1 for f in families if f.recipient_probe)
    n_sites = len(config.experiment.injection_sites)
    assert len(trials) == len(families) * 4 * n_sites * per_cell_site + (
        n_recipient_families * 4 * n_sites * per_cell_site_recipient
    )
    # Only the dative should carry them.
    assert {f.construction for f in families if f.recipient_probe} == {Construction.DATIVE}
    assert (Path(config.paths.results) / "trials.jsonl").exists()
    summary = analyze(config, trials)
    json.dumps(summary)  # must be JSON-serializable exactly as returned
    for figure in summary["figures"].values():
        assert Path(figure).stat().st_size > 0
    return summary


def test_binding_mode_recovers_large_pooled_score(tmp_path: Path) -> None:
    summary = _run(tmp_path, "binding")
    band_hi = summary["null_band"][1]
    assert summary["pooled"]["mean"] > 1.0  # dummy plants ~2.2 logits
    assert summary["pooled"]["mean"] > band_hi
    # The crossover signature: the gap shrinks from opposite sides, so BOTH
    # signed gap changes are negative.
    for sign in PushSign:
        assert summary["gap_change_by_sign"][sign.value]["mean"] < -1.0
    assert summary["neutral_strength_check"]["passes"] is True
    # Verdict: with the strength check passing, a pooled-significant d >= 0.5
    # effect classifies as at least suggestive; positive_binding additionally
    # needs per-construction Holm significance, unreachable at tiny-CI n.
    verdict = summary["verdict"]
    assert verdict["strength_check_passes"] is True
    assert verdict["pooled_significant"] is True
    assert verdict["outcome"] in ("positive_binding", "suggestive_not_conclusive")
    assert verdict["random_direction_in_band"] is True
    # d carries its own bootstrap CI, excluding zero here.
    assert summary["pooled"]["cohens_d_ci_lo"] > 0.0
    # Per-construction blocks exist for all four constructions.
    assert set(summary["constructions"]) == {
        "active_passive", "dative", "cleft", "relative_clause"
    }


def test_bag_mode_lands_inside_null_band(tmp_path: Path) -> None:
    summary = _run(tmp_path, "bag")
    band_lo, band_hi = summary["null_band"]
    mean = summary["pooled"]["mean"]
    assert abs(mean) < 0.15
    assert band_lo <= mean <= band_hi
    # A uniform log-odds push leaves each signed gap unchanged up to jitter.
    for sign in PushSign:
        assert abs(summary["gap_change_by_sign"][sign.value]["mean"]) < 0.15
    # The strength check still passes: "the edit works" is exactly what makes
    # a bag-mode null interpretable as a clean negative.
    assert summary["neutral_strength_check"]["passes"] is True
    # And the verdict says so: not significant, tight CI on d -> clean negative,
    # reported as a positive finding for the falsifiable claim.
    assert summary["verdict"]["outcome"] == "clean_negative"


def test_strength_check_requires_a_magnitude_not_just_a_sign() -> None:
    """A swap that moves the counterpart 0.3% -> 0.4% has not landed.

    Observed on Qwen3.6-27B: counterpart 0.00269 -> 0.00393 (+0.0012) with the
    entity essentially flat at 0.1851 -> 0.1809 (-0.0042). Both signs point the
    right way, so a sign-only rule returned `passes: true` -- and that pass is
    what licensed reading the primary's null as the concept-vs-role
    ADDRESSABILITY DISSOCIATION, the project's one positive claim.

    Meanwhile `calibrate_identity_alpha` reported an intervention-strength
    failure on the same model at min_prob_shift=0.05. The two are the same
    question asked twice and disagreed; this pins them together.

    Gemma-3-12B is the instructive contrast: the entity DROPS hard
    (0.2122 -> 0.0961) while the counterpart barely moves
    (0.00544 -> 0.01021). That is the swap damaging the entity readout without
    installing the counterpart -- consistent with the identity-overlap account
    from the Gemma pilot, and not a working swap either.
    """
    from jspace_binding.experiments.primary import _MIN_COUNTERPART_SHIFT

    def passes(counterpart_shift: float, entity_shift: float) -> bool:
        return bool(
            counterpart_shift >= _MIN_COUNTERPART_SHIFT and entity_shift < 0.0
        )

    # Qwen: nothing moved.
    assert passes(0.003931 - 0.002690, 0.180907 - 0.185074) is False
    # Gemma: entity damaged, counterpart never installed.
    assert passes(0.010207 - 0.005436, 0.096120 - 0.212159) is False
    # The dummy's planted swap, which genuinely propagates, must still pass.
    assert passes(0.55 - 0.01, 0.05 - 0.45) is True
    # A rise that clears the floor but with the entity going UP is still a
    # failure: the counterpart must displace the entity, not join it.
    assert passes(0.20, +0.05) is False


def test_push_gate_cannot_fail_the_null_it_exists_to_license(tmp_path: Path) -> None:
    """Non-circularity, on ground truth rather than a synthetic construction.

    The strength gate exists so a NULL binding score can be read as evidence
    about binding instead of a dead intervention. If the gate's statistic were
    entangled with the binding score, a null would fail its own gate and no
    null could ever be reported -- the gate would silently delete the
    project's whole class of findings.

    DummyModel's bag mode is exactly that case: a push that moves the readout
    (uniform sign*0.8) with no role structure behind it. The gate must see the
    movement while the binding score correctly reads ~0.

    This is not hypothetical. The first implementation used the "steering
    mirror", the difference of the two signed gap changes, on the reasoning
    that the binding score is their SUM and the two are orthogonal contrasts.
    Both gap changes are taken against the same natural gap, so a role-uniform
    push cancels completely: the mirror is identically zero in BOTH dummy
    modes, and gating on it turned this clean negative into
    uninterpretable_strength_failure.
    """
    summary = _run(tmp_path, "bag")
    check = summary["push_strength_check"]
    band_lo, band_hi = summary["null_band"]

    # The intervention demonstrably moved the readout ...
    assert check["available"] is True
    assert check["passes"] is True
    assert check["role_push_displacement"] >= check["min_push_displacement"]
    # ... while the binding score sits in the null band. Both at once is the
    # property; either alone proves nothing.
    assert band_lo <= summary["pooled"]["mean"] <= band_hi
    assert summary["verdict"]["outcome"] == "clean_negative"


def test_gemma_lre_push_arm_passes_where_the_swap_arm_fails() -> None:
    """Observed Gemma-3-12B LRE numbers, entity_token (runs/gemma3-12b-lre).

    The run this gate change rescues. IDENTITY_SWAP's alpha never calibrated,
    so the swap arm fails; the ROLE_PUSH arm calibrated cleanly and displaces
    the readout. Gating a push result on the swap arm made this run
    uninterpretable by construction.

    The margin is NOT comfortable and this test says so: 0.593 against 0.433
    for a direction fitted on PERMUTED labels, a ratio of 1.37. Most of the
    displacement is what any fitted direction of that norm achieves, not
    something role-specific, and the paper must say so rather than quoting
    0.593 against random_direction's 0.042 and implying 14x.
    """
    from jspace_binding.experiments.primary import (
        _MIN_COUNTERPART_SHIFT,
        _MIN_PUSH_DISPLACEMENT,
    )

    role, shuffled_label, absent, random_dir = 0.59259, 0.43322, 0.43314, 0.04159
    assert role >= _MIN_PUSH_DISPLACEMENT
    assert role > shuffled_label          # paired CI [0.153, 0.166], excludes zero
    assert role / shuffled_label < 1.5    # ... but only just; see docstring
    # Ranking the controls and taking the STRONGEST is load-bearing: two
    # controls sit near 0.433 while random sits at 0.042, so comparing against
    # random alone would inflate the margin by an order of magnitude.
    assert abs(shuffled_label - absent) < 0.01
    assert shuffled_label / random_dir > 10.0

    # The swap arm on the same run, for contrast: counterpart +0.00477 against
    # a 0.05 floor. Still failing, and still correctly reported as failing.
    assert _MIN_COUNTERPART_SHIFT > (0.010207 - 0.005436)


def test_significance_inside_the_null_band_is_not_a_small_effect() -> None:
    """A pooled mean INSIDE the control band must not be classified on effect
    size. p and Cohen's d both answer "how consistently does the push move the
    score", neither answers "compared to what" -- a strength-matched push of an
    ABSENT entity's direction moves it just as far.

    Pins the real Gemma-3-27B-IT pattern (runs/gemma3-27b-it-c128, c256): the
    push was Holm-significant in every construction with |d| up to 0.68 while
    sitting inside its own null band and within 20% of NULL_NON_PARTICIPANT --
    at c=256 the absent-entity push moved the score MORE than the real one.
    The old rule read that as significant_but_tiny, which implies a real effect
    that is merely small.
    """
    from jspace_binding.analysis.binding_score import ScoreTable
    from jspace_binding.experiments.primary import _verdict

    config = Config.from_yaml("configs/default.yaml")
    # c=256 numbers, verified by reanalyze_primary over the archived trials.
    pooled = {
        "n": 600, "mean": -0.1809, "p_perm": 1e-4,
        "cohens_d": -0.681, "cohens_d_ci_lo": -0.788, "cohens_d_ci_hi": -0.582,
    }
    table = ScoreTable(
        null_by_edit={
            "null_non_participant": [-0.2161] * 20,
            "random_direction": [-0.0035] * 20,
            "shuffled_label_direction": [-0.1329] * 20,
        },
    )
    verdict = _verdict(
        config, pooled, {}, {}, table, (-0.481, 0.154),
        push_check={"available": True, "passes": True},
        neutral_check={"available": True, "passes": False},
    )
    assert verdict["outcome"] == "indistinguishable_from_controls"
    assert verdict["clears_null_band"] is False
    assert verdict["strongest_control"] == "null_non_participant"
    # The real push moved the score LESS than the absent-entity control.
    assert verdict["vs_strongest_control_ratio"] < 1.0

    # Same significance and effect size, but clearing the band, is a real
    # (if small) effect and must still be classified on size.
    cleared = _verdict(
        config, {**pooled, "mean": 0.30, "cohens_d": 0.30,
                 "cohens_d_ci_lo": 0.2, "cohens_d_ci_hi": 0.4},
        {}, {}, table, (-0.481, 0.154),
        push_check={"available": True, "passes": True},
        neutral_check={"available": True, "passes": False},
    )
    assert cleared["outcome"] == "significant_but_tiny"
    assert cleared["clears_null_band"] is True
