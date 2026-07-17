"""Publication figures for the binding-score analysis.

Headless by design: the Agg backend is pinned before pyplot is imported so the
pipeline behaves identically on CI, laptops, and GPU boxes with no display.
Both functions write to `out_path` and close their figure — nothing is ever
shown interactively.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jspace_binding.analysis.binding_score import ScoreTable  # noqa: E402
from jspace_binding.types import (  # noqa: E402
    EditType,
    InjectionSite,
    ProbeKind,
    PushSign,
    Role,
    TrialResult,
)

# Matches ScoreTable.real keys: (construction.value, pair_id).
GroupKey = tuple[str, str]


def forest_plot(
    score_table: ScoreTable,
    stats_by_group: dict[GroupKey, dict[str, float]],
    null_band: tuple[float, float],
    out_path: str | Path,
) -> None:
    """One row per (construction x pair) group: mean BS with CI whiskers over
    the shaded control null band. The paper figure.

    `stats_by_group` values must carry "mean", "ci_lo", "ci_hi" (the shape
    experiments.primary.analyze builds). Per-family scores are overplotted as
    faint dots so a group mean cannot hide multimodality.
    """
    groups = [key for key in score_table.real if key in stats_by_group]
    fig, ax = plt.subplots(figsize=(7.0, max(2.5, 0.55 * len(groups) + 1.5)))
    band_lo, band_hi = null_band
    ax.axvspan(band_lo, band_hi, color="0.88", zorder=0, label="control null band")
    ax.axvline(0.0, color="0.5", linewidth=0.8, zorder=1)
    ys = np.arange(len(groups), dtype=float)[::-1]
    for y, key in zip(ys, groups, strict=True):
        block = stats_by_group[key]
        scores = score_table.real[key]
        ax.plot(scores, np.full(len(scores), y), ".", color="C0", alpha=0.35, zorder=2)
        ax.errorbar(
            [block["mean"]],
            [y],
            # Percentile CIs bracket the mean; clamp guards float round-off only.
            xerr=[
                [max(0.0, block["mean"] - block["ci_lo"])],
                [max(0.0, block["ci_hi"] - block["mean"])],
            ],
            fmt="o",
            color="C0",
            capsize=3,
            zorder=3,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{construction} | {pair_id}" for construction, pair_id in groups])
    ax.set_xlabel("binding score (log-odds, both-sign crossover)")
    ax.set_title("Role-filler binding score by construction and concept pair")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out_path)


def per_condition_plot(
    all_trials: list[TrialResult],
    entity_token: str,
    out_path: str | Path,
) -> None:
    """Raw agent-vs-patient P(entity) under each (edit type x push sign)
    condition — the undifferenced view behind the binding score (ROLE probe,
    FINAL_TOKEN site). This is the figure where the crossover is visible as a
    pattern: the toward-agent push lifting the patient bars, the
    toward-patient push dropping the agent bars.

    Only trials whose answer set contains `entity_token` contribute, so passing
    the full trial list plots the one concept pair that token belongs to.
    """
    rows = [
        t
        for t in all_trials
        if t.probe_kind is ProbeKind.ROLE
        and t.injection_site is InjectionSite.FINAL_TOKEN
        and entity_token in t.answer_probs
    ]
    if not rows:
        raise ValueError(f"no ROLE-probe final-token trials carry answer token {entity_token!r}")
    conditions = []
    for edit_type in EditType:
        for sign in (None, *PushSign):
            if any(t.edit_type is edit_type and t.push_sign is sign for t in rows):
                conditions.append((edit_type, sign))
    x = np.arange(len(conditions), dtype=float)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(conditions)), 4.0))
    for offset, role in ((-width / 2, Role.AGENT), (width / 2, Role.PATIENT)):
        means, spreads = [], []
        for edit_type, sign in conditions:
            values = [
                t.answer_probs[entity_token]
                for t in rows
                if t.edit_type is edit_type and t.push_sign is sign and t.role is role
            ]
            means.append(float(np.mean(values)))
            spreads.append(float(np.std(values)))
        ax.bar(x + offset, means, width, yerr=spreads, capsize=3, label=f"entity is {role.value}")
    ax.set_xticks(x)
    ax.set_xticklabels([_condition_label(e, s) for e, s in conditions], rotation=25, ha="right")
    ax.set_ylabel(f"P({entity_token}) at the ROLE probe")
    ax.set_title("Per-condition entity probability (final-token site)")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    _save(fig, out_path)


def _condition_label(edit_type: EditType, sign: PushSign | None) -> str:
    if sign is None:
        return edit_type.value
    arrow = "→agent" if sign is PushSign.TOWARD_AGENT else "→patient"
    return f"{edit_type.value} {arrow}"


def _save(fig: plt.Figure, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
