"""Fig 1 comparison: cross-day matrix at CCA d=15 (current) vs d=3 (proposed).

The CCA sweep (big_sweep_cca.py) showed only the top ~1-3 canonical dimensions
carry cross-day-shared signal; using all 15 drags in noise-aligned dimensions
and depresses off-diagonal decoding. This figure rebuilds the Fig 1 matrix at
d=3 and places it next to the current d=15 version.

No re-run is needed:
  * off-diagonal (cross-day) cells come from cca_sweep_long.csv at n_cca=d
    (locked config: bin=30, butter_o2, lag=0, Kalman, 0828 trial-41 excluded);
  * diagonal (within-day) cells are CCA-independent and reused verbatim from the
    main sweep (big_sweep_crossday_long.csv via filter_locked / pivot_matrix).
The d=15 off-diagonal cells from cca_sweep match the main sweep exactly, so the
two sources stitch cleanly.

Rows = target (velocity, position); columns = d=15 then d=3. Colormap vmax is
shared within a row so the brightening at d=3 is directly comparable.

Writes Results/generalization/figures/fig1_ccadim_compare.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS))

from plotting_common import (
    ALL_SESSIONS, CMAP_PRIMARY, METRIC_LABEL, SECONDARY_TARGET,
    config_caption, draw_diagonal_frames, draw_r1r2_split, ensure_fig_dir,
    filter_locked, load_sweep, pivot_matrix, set_session_ticks, shared_vmax,
)

REPO_ROOT = _THIS.parents[1]
CCA_CSV = REPO_ROOT / "Results" / "generalization" / "cca_sweep_long.csv"


def matrix_at_dim(main_df, cca_df, target_mode, d):
    """Stitch a 15x15 matrix: diagonal from main sweep, off-diagonal at CCA dim d."""
    # Diagonal (within-day, CCA-independent) from the locked main sweep.
    diag_mat = pivot_matrix(filter_locked(main_df, target_mode=target_mode)).values.copy()

    # Off-diagonal from the CCA sweep at n_cca = d.
    sub = cca_df[(cca_df["metric"] == "decode")
                 & (cca_df["target_mode"] == target_mode)
                 & (cca_df["n_cca"] == d)]
    off = (sub.pivot_table(index="train_session", columns="test_session", values="corr")
           .reindex(index=ALL_SESSIONS, columns=ALL_SESSIONS).values)

    out = diag_mat.copy()
    mask = ~np.eye(len(ALL_SESSIONS), dtype=bool)
    out[mask] = off[mask]
    return out


def panel(ax, mat, title, vmin, vmax):
    im = ax.imshow(mat, cmap=CMAP_PRIMARY, vmin=vmin, vmax=vmax, aspect="equal")
    draw_diagonal_frames(ax)
    draw_r1r2_split(ax)
    set_session_ticks(ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Test session")
    ax.set_ylabel("Train session")
    return im


def off_diag_mean(mat):
    m = ~np.eye(mat.shape[0], dtype=bool)
    return float(np.nanmean(mat[m]))


def main():
    out_dir = ensure_fig_dir()
    main_df = load_sweep()
    cca_df = pd.read_csv(CCA_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for r, (tm, label) in enumerate([("relative_velocity", "velocity"),
                                      (SECONDARY_TARGET, "position")]):
        m15 = matrix_at_dim(main_df, cca_df, tm, 15)
        m3 = matrix_at_dim(main_df, cca_df, tm, 3)
        vmax = shared_vmax(m15, m3, quantile=0.99)
        im = panel(axes[r, 0], m15,
                   f"{label} · CCA d=15 (current)\ncross-day mean = {off_diag_mean(m15):.3f}",
                   0.0, vmax)
        panel(axes[r, 1], m3,
              f"{label} · CCA d=3 (proposed)\ncross-day mean = {off_diag_mean(m3):.3f}",
              0.0, vmax)
        cbar = fig.colorbar(im, ax=[axes[r, 0], axes[r, 1]], shrink=0.8, pad=0.02)
        cbar.set_label(METRIC_LABEL)

    fig.suptitle(
        "Fig 1 cross-day matrix: CCA d=15 vs d=3  ·  diagonal (within-day) is "
        "CCA-independent and identical\n"
        f"{config_caption()} (off-diagonal cells differ only in # CCA dims kept)",
        fontsize=11, y=1.01,
    )
    out = out_dir / "fig1_ccadim_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")

    for tm, label in [("relative_velocity", "velocity"), (SECONDARY_TARGET, "position")]:
        m15 = matrix_at_dim(main_df, cca_df, tm, 15)
        m3 = matrix_at_dim(main_df, cca_df, tm, 3)
        print(f"{label:9s}: cross-day mean  d15={off_diag_mean(m15):.3f}  "
              f"d3={off_diag_mean(m3):.3f}  (+{off_diag_mean(m3)-off_diag_mean(m15):+.3f})")


if __name__ == "__main__":
    main()
