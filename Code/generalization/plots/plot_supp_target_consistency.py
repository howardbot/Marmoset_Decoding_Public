"""S3: Target consistency -- velocity vs position scatter (Kalman, locked).

Single panel: each point is one off-diagonal pair (train_session, test_session)
at LOCKED_CONFIG, with x = M2 when decoding velocity, y = M2 when decoding
position. Points are color-coded by pair category (R1->R1 / R1->R2 / R2->R1 /
R2->R2) so the reader can see that the R1 vs R2 structure shows up in both
targets.

Spearman rho between vel and pos M2 quantifies the cross-target agreement.

Writes ``Results/workflows/generalization/figures/fig_supp_target_consistency.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS))

from plotting_common import (
    ALL_SESSIONS, PAIR_COLORS, SECONDARY_TARGET, config_caption,
    ensure_fig_dir, filter_locked, load_sweep, pair_category, pivot_matrix,
)


def main():
    out_dir = ensure_fig_dir()
    df = load_sweep()

    vel = pivot_matrix(filter_locked(df))                                   # vel locked
    pos = pivot_matrix(filter_locked(df, target_mode=SECONDARY_TARGET))     # pos locked

    n = len(ALL_SESSIONS)
    records = []
    for i, tr in enumerate(ALL_SESSIONS):
        for j, te in enumerate(ALL_SESSIONS):
            if i == j:
                continue  # off-diagonal only
            v = vel.iat[i, j]; p = pos.iat[i, j]
            if not (np.isfinite(v) and np.isfinite(p)):
                continue
            records.append((v, p, pair_category(tr, te)))

    pts = np.array([(r[0], r[1]) for r in records])
    cats = [r[2] for r in records]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for cat, color in PAIR_COLORS.items():
        m = np.array([c == cat for c in cats])
        if not m.any():
            continue
        ax.scatter(pts[m, 0], pts[m, 1], s=42, alpha=0.7, color=color,
                   label=f"{cat}  (n={m.sum()})", edgecolors="white",
                   linewidth=0.6)
    lim_lo = min(pts.min(), 0.0) - 0.03
    lim_hi = pts.max() + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("velocity corr")
    ax.set_ylabel("position corr")
    ax.set_title("Target consistency: cross-day pairs across velocity and position",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)

    fig.suptitle(config_caption(), fontsize=9, y=0.98, color="gray")
    out = out_dir / "fig_supp_target_consistency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
