"""F3: time-drift control for the R1->R2 forward drop.

Question: is the forward (R1->R2) drop just natural representational drift over
calendar time, since R1->R2 pairs span more days than R1->R1 pairs?

Design (corr vs calendar day-gap):
  * gray points  : R1->R1 pairs -> define the natural within-epoch drift trend.
                   A linear fit is drawn solid over the observed R1 range and
                   dashed (extrapolated) beyond it, with the extrapolation
                   region shaded to flag that it is an extrapolation.
  * red points   : R1->R2 (forward)
  * blue points  : R2->R1 (reverse)

Primary argument (needs no extrapolation): forward and reverse use the SAME
session pairs with train/test swapped, so they sit at IDENTICAL day-gaps. If
forward falls below reverse at matched gaps, calendar time is controlled by
construction and cannot explain the drop.

Two panels: Kalman/velocity and Kalman/position at LOCKED_CONFIG.

Writes ``Results/workflows/generalization/figures/fig3_time_drift.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS))

from plotting_common import (
    ALL_SESSIONS, METRIC_LABEL, PAIR_COLORS, SECONDARY_TARGET, config_caption,
    day_gap, ensure_fig_dir, filter_locked, load_sweep, pair_category,
    pivot_matrix,
)

R1R1_COLOR = "#9aa0a6"
FORWARD_COLOR = PAIR_COLORS["R1->R2"]
REVERSE_COLOR = PAIR_COLORS["R2->R1"]


def gather(mat):
    """Collect (day_gap, corr, category) for every off-diagonal pair."""
    recs = []
    for tr in ALL_SESSIONS:
        for te in ALL_SESSIONS:
            if tr == te:
                continue
            v = mat.at[tr, te]
            if np.isfinite(v):
                recs.append((day_gap(tr, te), float(v), pair_category(tr, te)))
    return recs


def panel(ax, mat, title):
    recs = gather(mat)
    gaps = np.array([r[0] for r in recs])
    vals = np.array([r[1] for r in recs])
    cats = np.array([r[2] for r in recs])

    r1r1 = cats == "R1->R1"
    fwd = cats == "R1->R2"
    rev = cats == "R2->R1"

    # Natural-drift linear fit on R1->R1 only.
    g1, v1 = gaps[r1r1], vals[r1r1]
    slope, intercept = np.polyfit(g1, v1, 1)
    g1_max = g1.max()
    xs_obs = np.linspace(0, g1_max, 50)
    xs_ext = np.linspace(g1_max, gaps.max() + 1, 50)
    ax.plot(xs_obs, slope * xs_obs + intercept, color="black", linewidth=2,
            label="R1→R1 drift fit")
    ax.plot(xs_ext, slope * xs_ext + intercept, color="black", linewidth=2,
            linestyle="--", label="extrapolation")
    # Shade the extrapolation region.
    ax.axvspan(g1_max, gaps.max() + 1, color="#f0f0f0", zorder=0)

    # Points.
    ax.scatter(gaps[r1r1], vals[r1r1], s=20, color=R1R1_COLOR, alpha=0.45,
               edgecolors="none", label="R1→R1 (natural drift)", zorder=2)
    ax.scatter(gaps[fwd], vals[fwd], s=45, color=FORWARD_COLOR, alpha=0.85,
               edgecolors="white", linewidth=0.5, label="R1→R2 (forward)", zorder=4)
    ax.scatter(gaps[rev], vals[rev], s=45, color=REVERSE_COLOR, alpha=0.85,
               edgecolors="white", linewidth=0.5, label="R2→R1 (reverse)", zorder=4)

    ax.set_xlabel("calendar day-gap between train and test session")
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)


def main():
    out_dir = ensure_fig_dir()
    df = load_sweep()

    mat_vel = pivot_matrix(filter_locked(df))
    mat_pos = pivot_matrix(filter_locked(df, target_mode=SECONDARY_TARGET))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panel(axes[0], mat_vel, "Kalman · velocity")
    panel(axes[1], mat_pos, "Kalman · position")

    fig.suptitle(
        "Time-drift control — at matched day-gaps reverse (R2→R1) exceeds "
        "forward (R1→R2); same session pairs, so calendar time is controlled\n"
        f"{config_caption()}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "fig3_time_drift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
