"""F2: R1 <-> R2 transfer asymmetry (the Aim 3 retrograde-interference figure).

Deviation-from-baseline design. The R1->R1 off-diagonal mean is the natural
within-epoch transfer baseline (zero line). We then show how each cross-epoch
direction deviates from it:

  R1->R2 (forward)  -> bar drops below zero  (representation changed; old
                       R1 decoder can't read R2 activity)
  R2->R1 (reverse)  -> bar sits at ~zero     (R2 decoder still reads R1;
                       no loss in the reverse direction)

Individual (pair - baseline) deviations are overlaid as jittered points so the
spread is visible. The asymmetry between the two bars -- not the absolute
height -- is the signal: a symmetric epoch difference would drop both.

Two panels: Kalman / velocity and Kalman / position at LOCKED_CONFIG.

Writes ``Results/workflows/generalization/figures/fig2_r1r2_asymmetry.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS))

from plotting_common import (
    METRIC_LABEL, PAIR_COLORS, SECONDARY_TARGET, block_mean, config_caption,
    ensure_fig_dir, filter_locked, forward_reverse_pairs, load_sweep,
    pivot_matrix,
)

FORWARD_COLOR = PAIR_COLORS["R1->R2"]   # red
REVERSE_COLOR = PAIR_COLORS["R2->R1"]   # blue


def panel(ax, mat, title):
    pairs = forward_reverse_pairs(mat)
    fwd = np.array([p[2] for p in pairs])
    rev = np.array([p[3] for p in pairs])
    baseline = block_mean(mat, "R1", "R1")

    fwd_dev = fwd - baseline
    rev_dev = rev - baseline
    heights = [fwd_dev.mean(), rev_dev.mean()]
    abs_means = [fwd.mean(), rev.mean()]
    devs = [fwd_dev, rev_dev]
    colors = [FORWARD_COLOR, REVERSE_COLOR]
    xlabels = ["R1 → R2\n(forward)", "R2 → R1\n(reverse)"]

    # Baseline = zero line.
    ax.axhline(0, color="black", linewidth=1.4)
    ax.text(1.62, 0, f"R1→R1 baseline\n({baseline:.2f})", va="center",
            ha="left", fontsize=8, color="#444")

    rng = np.random.default_rng(0)
    for x, (h, dev, c) in enumerate(zip(heights, devs, colors)):
        ax.bar(x, h, width=0.55, color=c, alpha=0.55, zorder=1)
        jit = rng.normal(x, 0.07, size=len(dev))
        ax.scatter(jit, dev, s=22, color=c, alpha=0.8, edgecolors="white",
                   linewidth=0.4, zorder=3)
        # Absolute-value annotation above/below the bar.
        va = "bottom" if h >= 0 else "top"
        off = 0.004 if h >= 0 else -0.004
        ax.text(x, h + off, f"mean {abs_means[x]:.2f}", ha="center", va=va,
                fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_xlim(-0.6, 2.0)
    ax.set_ylabel(f"{METRIC_LABEL}  −  baseline")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)


def main():
    out_dir = ensure_fig_dir()
    df = load_sweep()

    mat_vel = pivot_matrix(filter_locked(df))
    mat_pos = pivot_matrix(filter_locked(df, target_mode=SECONDARY_TARGET))

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=False)
    panel(axes[0], mat_vel, "Kalman · velocity")
    panel(axes[1], mat_pos, "Kalman · position")

    fig.suptitle(
        "R1 ↔ R2 transfer asymmetry — forward (R1→R2) drops below baseline, "
        "reverse (R2→R1) does not\n"
        f"{config_caption()}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "fig2_r1r2_asymmetry.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
