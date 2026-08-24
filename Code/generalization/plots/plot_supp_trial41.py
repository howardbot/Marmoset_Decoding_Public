"""Trial-41 outlier sanity check: include vs exclude vs delta.

Three-panel figure proving that the decision to drop 0828 trial 41 affects
only the 0828 row/column, not the rest of the matrix:

  Left   : 15x15 with trial 41 INCLUDED   -- 0828 row/col collapse to ~0
  Middle : 15x15 with trial 41 EXCLUDED   -- matrix restored, locked baseline
  Right  : Delta matrix  (exclude - include) -- diverging cmap; lights up only
           the 0828 row/col, confirming the fix is localized

Uses Kalman / relative_velocity (LOCKED_CONFIG primary) for the comparison.

Writes ``Results/workflows/generalization/figures/fig_supp_trial41_outlier.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS))

from plotting_common import (
    CMAP_DIVERGING, CMAP_PRIMARY, METRIC_LABEL, R2_BOUNDARY_INDEX,
    config_caption, draw_diagonal_frames, draw_r1r2_split, ensure_fig_dir,
    filter_locked, load_sweep, pivot_matrix, set_session_ticks, shared_vmax,
)


def heat_panel(ax, mat, title, cmap, vmin, vmax):
    im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    draw_diagonal_frames(ax)
    draw_r1r2_split(ax)
    set_session_ticks(ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Test session")
    ax.set_ylabel("Train session")
    return im


def main():
    out_dir = ensure_fig_dir()
    df = load_sweep()

    mat_inc = pivot_matrix(filter_locked(df, outlier_mode="include"))
    mat_exc = pivot_matrix(filter_locked(df))  # exclude is the default
    delta = mat_exc.values - mat_inc.values

    vmax = shared_vmax(mat_inc.values, mat_exc.values, quantile=0.99)
    # symmetric vlim for delta
    dlim = float(np.nanmax(np.abs(delta)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im0 = heat_panel(axes[0], mat_inc,
                     f"INCLUDE trial 41   diag={np.diag(mat_inc.values).mean():.2f}, "
                     f"0828-row={np.nanmean(mat_inc.values[R2_BOUNDARY_INDEX, :]):.2f}",
                     CMAP_PRIMARY, 0.0, vmax)
    im1 = heat_panel(axes[1], mat_exc,
                     f"EXCLUDE trial 41   diag={np.diag(mat_exc.values).mean():.2f}, "
                     f"0828-row={np.nanmean(mat_exc.values[R2_BOUNDARY_INDEX, :]):.2f}",
                     CMAP_PRIMARY, 0.0, vmax)
    fig.colorbar(im1, ax=[axes[0], axes[1]], shrink=0.85, pad=0.02,
                 label=METRIC_LABEL)

    im2 = heat_panel(axes[2], type(mat_inc)(delta, index=mat_inc.index, columns=mat_inc.columns),
                     "Δ matrix (exclude − include)\nlocalized to 0828 row/col",
                     CMAP_DIVERGING, -dlim, dlim)
    fig.colorbar(im2, ax=axes[2], shrink=0.85, pad=0.02,
                 label="Δ corr (exclude − include)")

    fig.suptitle(
        "0828 trial 41 outlier sanity check  ·  single trial drags the row/col;\n"
        f"removing it does not affect the rest of the matrix.   {config_caption()}",
        fontsize=11, y=1.02,
    )

    out = out_dir / "fig_supp_trial41_outlier.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
