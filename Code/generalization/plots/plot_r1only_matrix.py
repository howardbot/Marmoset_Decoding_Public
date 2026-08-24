"""R1-only cross-day decoding matrix (velocity vs position) for the slide
'Position is better decoded than velocity'. R1->R1 pairs never involve 0828, so this is
immune to the outlier_mode filtering issue; we simply slice the R1 block out of the locked
Kalman matrix and re-scale the colormap to the R1 values so the structure is visible.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS.parent)); sys.path.insert(0, str(_THIS))

from plotting_common import (
    SECONDARY_TARGET, CMAP_PRIMARY, METRIC_LABEL, SESSIONS_R1,
    config_caption, filter_locked, load_sweep, pivot_matrix, ensure_fig_dir,
)

R1_LABELS = [s.split("_")[0][-4:] for s in SESSIONS_R1]   # e.g. 0731 ... 0813


def r1_block(target=None):
    df = filter_locked(load_sweep()) if target is None else filter_locked(load_sweep(), target_mode=target)
    mat = pivot_matrix(df).reindex(index=SESSIONS_R1, columns=SESSIONS_R1)
    return mat.values.astype(float)


def diag_frames(ax, n, color="#d62728", lw=1.6):
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor=color, lw=lw, zorder=5))


def panel(ax, mat, title, vmin, vmax):
    im = ax.imshow(mat, cmap=CMAP_PRIMARY, vmin=vmin, vmax=vmax, aspect="equal")
    diag_frames(ax, mat.shape[0])
    ax.set_xticks(range(len(R1_LABELS))); ax.set_xticklabels(R1_LABELS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(R1_LABELS))); ax.set_yticklabels(R1_LABELS, fontsize=7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Test session (R1)"); ax.set_ylabel("Train session (R1)")
    return im


def main():
    out_dir = ensure_fig_dir()
    vel = r1_block(None)                 # relative_velocity (primary)
    pos = r1_block(SECONDARY_TARGET)     # relative_position
    print("R1 velocity: mean %.3f  diag-mean %.3f" % (np.nanmean(vel), np.nanmean(np.diag(vel))))
    print("R1 position: mean %.3f  diag-mean %.3f" % (np.nanmean(pos), np.nanmean(np.diag(pos))))

    vmax = float(np.nanpercentile(np.concatenate([vel.ravel(), pos.ravel()]), 99))
    vmin = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    panel(axes[0], vel, "Kalman · velocity", vmin, vmax)
    im = panel(axes[1], pos, "Kalman · position", vmin, vmax)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02); cbar.set_label(METRIC_LABEL)
    fig.suptitle("R1 within-epoch cross-day decoding (R1 generalizations) — position is brighter",
                 fontsize=12, y=1.01)
    out = out_dir / "fig_r1only_crossday_matrix.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
