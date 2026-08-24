"""S2: Decoder consistency -- Wiener heatmaps + Kalman vs Wiener scatter.

Three-panel supplement demonstrating that the cross-day R1/R2 pattern is not
an artefact of the Kalman decoder.

  Left   : Wiener / relative_velocity (15x15 heatmap)
  Middle : Wiener / relative_position (15x15 heatmap)
  Right  : Per-pair scatter of Kalman M2 vs Wiener M2 (across both targets),
           with y=x reference and Spearman rho. Same locked config; only
           decoder differs.

Writes ``Results/generalization/figures/fig_supp_decoder_consistency.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS))

from plotting_common import (
    ALL_SESSIONS, CMAP_PRIMARY, METRIC_LABEL, SECONDARY_TARGET,
    config_caption, draw_diagonal_frames, draw_r1r2_split, ensure_fig_dir,
    filter_locked, load_sweep, pivot_matrix, set_session_ticks, shared_vmax,
)


def plot_heatmap_panel(ax, mat, title, vmin, vmax):
    im = ax.imshow(mat.values, cmap=CMAP_PRIMARY, vmin=vmin, vmax=vmax, aspect="equal")
    draw_diagonal_frames(ax)
    draw_r1r2_split(ax)
    set_session_ticks(ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Test session")
    ax.set_ylabel("Train session")
    return im


def main():
    out_dir = ensure_fig_dir()
    df = load_sweep()

    # Wiener heatmaps at locked config (history=50).
    mat_w_vel = pivot_matrix(filter_locked(df, decoder="wiener"))
    mat_w_pos = pivot_matrix(filter_locked(df, decoder="wiener",
                                          target_mode=SECONDARY_TARGET))
    vmax = shared_vmax(mat_w_vel.values, mat_w_pos.values, quantile=0.99)

    # Scatter: pair Kalman vs Wiener at locked config for both targets.
    n = len(ALL_SESSIONS)
    off_mask = ~np.eye(n, dtype=bool)
    points = []
    for tgt in ("relative_velocity", SECONDARY_TARGET):
        k = pivot_matrix(filter_locked(df, target_mode=tgt))
        w = pivot_matrix(filter_locked(df, decoder="wiener", target_mode=tgt))
        for i in range(n):
            for j in range(n):
                if i == j or not np.isfinite(k.iat[i, j]) or not np.isfinite(w.iat[i, j]):
                    continue
                points.append((k.iat[i, j], w.iat[i, j], tgt))
    pts = np.array([(p[0], p[1]) for p in points])
    labels = [p[2] for p in points]

    fig = plt.figure(figsize=(18, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.30)

    ax0 = fig.add_subplot(gs[0])
    plot_heatmap_panel(ax0, mat_w_vel,
                       f"Wiener · velocity   "
                       f"diag={np.diag(mat_w_vel.values).mean():.2f}, "
                       f"off={np.nanmean(mat_w_vel.values[off_mask]):.2f}",
                       0.0, vmax)
    ax1 = fig.add_subplot(gs[1])
    im1 = plot_heatmap_panel(ax1, mat_w_pos,
                       f"Wiener · position   "
                       f"diag={np.diag(mat_w_pos.values).mean():.2f}, "
                       f"off={np.nanmean(mat_w_pos.values[off_mask]):.2f}",
                       0.0, vmax)
    cbar = fig.colorbar(im1, ax=[ax0, ax1], shrink=0.85, pad=0.02)
    cbar.set_label(METRIC_LABEL)

    ax2 = fig.add_subplot(gs[2])
    color_for = {"relative_velocity": "#e74c3c", SECONDARY_TARGET: "#3498db"}
    for tgt in ("relative_velocity", SECONDARY_TARGET):
        m = np.array([l == tgt for l in labels])
        ax2.scatter(pts[m, 0], pts[m, 1], s=22, alpha=0.55,
                    color=color_for[tgt], label=tgt, edgecolors="none")
    lim_lo = min(pts.min(), 0.0) - 0.05
    lim_hi = pts.max() + 0.05
    ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1, alpha=0.6, label="y = x")
    ax2.set_xlim(lim_lo, lim_hi)
    ax2.set_ylim(lim_lo, lim_hi)
    ax2.set_xlabel("Kalman — per-pair corr")
    ax2.set_ylabel("Wiener — per-pair corr")
    ax2.set_title("Per-pair decoder agreement (off-diagonal)", fontsize=11)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Decoder consistency: Wiener reproduces the Kalman pattern\n"
        f"{config_caption(decoder='wiener')}",
        fontsize=11, y=1.02,
    )
    out = out_dir / "fig_supp_decoder_consistency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
