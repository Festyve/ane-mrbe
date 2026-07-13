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
    ax.set_xlabel("binding score (difference-in-differences)")
    ax.set_title("Role-filler binding score by construction and concept pair")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out_path)


def per_condition_plot(
    all_trials: list[TrialResult],
    target_token: str,
    out_path: str | Path,
) -> None:
    """Raw agent-vs-patient P(target) under each edit type — the undifferenced
    view behind the binding score (ROLE probe, FINAL_TOKEN site).

    Only trials whose answer set contains `target_token` contribute, so passing
    the full trial list plots the one concept pair that token belongs to.
    """
    rows = [
        t
        for t in all_trials
        if t.probe_kind is ProbeKind.ROLE
        and t.injection_site is InjectionSite.FINAL_TOKEN
        and target_token in t.answer_probs
    ]
    if not rows:
        raise ValueError(f"no ROLE-probe final-token trials carry answer token {target_token!r}")
    edit_types = [e for e in EditType if any(t.edit_type is e for t in rows)]
    x = np.arange(len(edit_types), dtype=float)
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for offset, role in ((-width / 2, Role.AGENT), (width / 2, Role.PATIENT)):
        means, spreads = [], []
        for edit_type in edit_types:
            values = [
                t.answer_probs[target_token]
                for t in rows
                if t.edit_type is edit_type and t.role is role
            ]
            means.append(float(np.mean(values)))
            spreads.append(float(np.std(values)))
        ax.bar(x + offset, means, width, yerr=spreads, capsize=3, label=f"target is {role.value}")
    ax.set_xticks(x)
    ax.set_xticklabels([e.value for e in edit_types], rotation=15, ha="right")
    ax.set_ylabel(f"P({target_token}) at the ROLE probe")
    ax.set_title("Per-condition target probability (final-token site)")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    _save(fig, out_path)


def _save(fig: plt.Figure, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
