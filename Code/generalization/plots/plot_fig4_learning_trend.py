"""F4: learning trend across R1 (the static task being learned).

This is Sami's (d): within-day decoding (the matrix diagonal) for each R1 day,
with an ordinary-least-squares linear regression and its significance.

  velocity | position   (R1 days only; R2 is a different task)

Each bar = one day's same-day decoding. The line is the OLS fit; the legend
reports R^2 (fraction of day-to-day variance the trend explains) and the
p-value (probability of a trend this strong under no real relationship).

Writes ``Results/generalization/figures/fig4_learning_trend.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS))

from plotting_common import (
    METRIC_LABEL, SECONDARY_TARGET, SESSIONS_R1, config_caption,
    ensure_fig_dir, filter_locked, load_sweep, pivot_matrix,
)

R1_LABELS = [s.split("_")[0][-4:] for s in SESSIONS_R1]
BAR_COLOR = "#5a7d9a"
LINE_COLOR = "#c0392b"


def within_day_r1(mat):
    """Within-day (diagonal) corr for each R1 session, in chronological order."""
    return np.array([mat.at[s, s] for s in SESSIONS_R1])


def regression_panel(ax, values, title):
    """Bars = per-day within-day corr; line = OLS fit; legend = R^2 + p."""
    x = np.arange(len(SESSIONS_R1))
    ax.bar(x, values, width=0.65, color=BAR_COLOR, alpha=0.8, zorder=2)

    good = np.isfinite(values)
    slope, intercept = np.polyfit(x[good], values[good], 1)
    r, p = pearsonr(x[good], values[good])
    r2 = r ** 2
    sig = " *" if p < 0.05 else ""
    ax.plot(x, slope * x + intercept, color=LINE_COLOR, linewidth=2.2,
            zorder=3, label=f"OLS fit  R²={r2:.2f}, p={p:.3f}{sig}")

    ax.set_xticks(x)
    ax.set_xticklabels(R1_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, zorder=1)


def main():
    out_dir = ensure_fig_dir()
    df = load_sweep()

    mat_vel = pivot_matrix(filter_locked(df))
    mat_pos = pivot_matrix(filter_locked(df, target_mode=SECONDARY_TARGET))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    regression_panel(axes[0], within_day_r1(mat_vel), "within-day · velocity")
    regression_panel(axes[1], within_day_r1(mat_pos), "within-day · position")

    fig.suptitle(
        "Learning trend across R1 (static task) — within-day decoding vs session\n"
        f"{config_caption()}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_dir / "fig4_learning_trend.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
