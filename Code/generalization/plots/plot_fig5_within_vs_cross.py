"""F5: within-day vs cross-day generalization drop, per session.

For each session (chronological, R1 then R2) we show two bars:
  within  : diagonal cell (train = test = this session, 5-fold CV)
  cross   : mean of this session's row off-diagonal cells
            (train = this session, test = every other session)

The gap between the two bars is the "generalization drop" for that day. A thin
vertical divider separates R1 from R2 so day-level drops can be compared within
and across epochs.

Two panels: Kalman / velocity and Kalman / position at LOCKED_CONFIG.

Writes ``Results/generalization/figures/fig5_within_vs_cross.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS))

from plotting_common import (
    ALL_SESSIONS, METRIC_LABEL, R2_BOUNDARY_INDEX, SECONDARY_TARGET,
    SESSION_LABELS, config_caption, ensure_fig_dir, filter_locked, load_sweep,
    pivot_matrix,
)

WITHIN_COLOR = "#2c3e50"   # dark — own-day ceiling
CROSS_COLOR = "#95c8d8"    # light — generalization


def within_and_cross(mat):
    """Return (within[], cross[]) per session in ALL_SESSIONS order."""
    n = len(ALL_SESSIONS)
    v = mat.values
    within = np.diag(v).astype(float)
    cross = np.array([
        np.nanmean([v[i, j] for j in range(n) if j != i]) for i in range(n)
    ])
    return within, cross


def panel(ax, mat, title):
    within, cross = within_and_cross(mat)
    n = len(ALL_SESSIONS)
    x = np.arange(n)
    w = 0.4
    ymax = max(np.nanmax(within), np.nanmax(cross))

    # Faint epoch background bands so the R1/R2 split is clear without crowding.
    ax.axvspan(-0.6, R2_BOUNDARY_INDEX - 0.5, color="#f2f4f5", zorder=0)
    ax.axvspan(R2_BOUNDARY_INDEX - 0.5, n - 0.4, color="#fdf1e6", zorder=0)

    ax.bar(x - w / 2, within, width=w, color=WITHIN_COLOR, label="within-day", zorder=2)
    ax.bar(x + w / 2, cross, width=w, color=CROSS_COLOR, label="cross-day (mean)", zorder=2)
    ax.axvline(R2_BOUNDARY_INDEX - 0.5, color="black", linewidth=1.2, linestyle="--", zorder=3)

    # Headroom so epoch labels and legend don't collide with the bars.
    ax.set_ylim(0, ymax * 1.28)
    ax.set_xlim(-0.6, n - 0.4)
    ax.text((R2_BOUNDARY_INDEX - 1) / 2, ymax * 1.17, "R1 (static)",
            ha="center", fontsize=10, color="#555", fontweight="bold")
    ax.text((R2_BOUNDARY_INDEX + n - 1) / 2, ymax * 1.17, "R2 (interf.)",
            ha="center", fontsize=10, color="#a0522d", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(SESSION_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, zorder=1)


def main():
    out_dir = ensure_fig_dir()
    df = load_sweep()

    mat_vel = pivot_matrix(filter_locked(df))
    mat_pos = pivot_matrix(filter_locked(df, target_mode=SECONDARY_TARGET))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    panel(axes[0], mat_vel, "Kalman · velocity")
    panel(axes[1], mat_pos, "Kalman · position")

    fig.suptitle(
        "Within-day vs cross-day decoding, per session  ·  gap = generalization drop\n"
        f"{config_caption()}",
        fontsize=11, y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out_dir / "fig5_within_vs_cross.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
