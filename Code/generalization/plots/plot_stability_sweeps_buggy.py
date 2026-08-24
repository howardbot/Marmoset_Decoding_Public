"""UNFIXED (buggy) reproduction of fig_stability_sweeps for before/after comparison.

The bug: hard-filter outlier_mode == "exclude" with NO include fallback. Because only
0828-involving cross-epoch pairs have an 'exclude' row, this silently drops all R1<->0829
and R1<->0830 pairs -> the cross-epoch panels represent 0828 ONLY, despite an n(R2)=3 label.
Panels 4-6 (from dimension_sweep_long, no outlier_mode) are unaffected; only panels 1-3 change.
Output: fig_stability_sweeps_buggy.png  (does NOT overwrite the fixed figure).
"""
from __future__ import annotations
import pathlib
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from plotting_common import filter_locked

RES = REPO / "Results" / "workflows" / "decoder_benchmarks"
OUT = RES / "generalization" / "figures" / "fig_stability_sweeps_buggy.png"
COL = {"R1->R2": "#e74c3c", "R2->R1": "#3498db", "R1->R1": "#9aa0a6"}

big = pd.read_csv(RES / "generalization" / "big_sweep_crossday_long.csv")
dim = pd.read_csv(RES / "manifold_geometry" / "dimension_sweep_long.csv")
ep = lambda s: "R2" if any(x in s for x in ("20250828", "20250829", "20250830")) else "R1"
big = big[big.train_session != big.test_session].copy()
big["pair"] = big.train_session.map(ep) + "->" + big.test_session.map(ep)


def buggy(df, **kw):
    """The bug: take the locked set then keep ONLY exclude rows (no include fallback)."""
    f = filter_locked(df, **kw)
    return f[f.outlier_mode == "exclude"].copy()


def line_panel(ax, df, xcol, ycol, title, xlabel, cats=("R1->R1", "R2->R1", "R1->R2")):
    for pc in cats:
        g = df[df.pair_cat == pc].groupby(xcol)[ycol].mean().reset_index() if "pair_cat" in df \
            else df[df.pair == pc].groupby(xcol)[ycol].mean().reset_index()
        if g.empty:
            continue
        ax.plot(g[xcol], g[ycol], "-o", color=COL[pc], label=pc, ms=4, lw=2)
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel); ax.set_ylabel("cross-day decode corr")
    ax.grid(alpha=.3)


def cat_panel(ax, df, xcol, ycol, title, order):
    xs = list(order)
    for pc in ("R1->R1", "R2->R1", "R1->R2"):
        sub = df[(df.pair == pc)] if "pair" in df else df[df.pair_cat == pc]
        if sub.empty:
            continue
        m = sub.groupby(xcol)[ycol].mean()
        ax.plot(range(len(xs)), [m.get(v, np.nan) for v in xs], "-o", color=COL[pc], label=pc, ms=7, lw=2)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
    ax.set_title(title, fontsize=10); ax.set_ylabel("cross-day decode corr"); ax.grid(alpha=.3, axis="y")


fig, ax = plt.subplots(2, 3, figsize=(16, 9))

# 1-3 use the BUGGY exclude-only filter
s = pd.concat([buggy(big, target_mode="relative_position", decoder="kalman", lag_ms=lag)
               for lag in sorted(big.lag_ms.unique())], ignore_index=True)
line_panel(ax[0, 0], s, "lag_ms", "M2_mean", "(1) lag sweep  [position, kalman]  BUGGY", "lag (ms)")

s = pd.concat([buggy(big, target_mode="relative_position", lag_ms=0, decoder=decoder,
                     **({"history_ms": 50} if decoder == "wiener" else {}))
               for decoder in ("kalman", "wiener")], ignore_index=True)
cat_panel(ax[0, 1], s, "decoder", "M2_mean", "(2) decoder  [position, lag 0]  BUGGY", ["kalman", "wiener"])

s = pd.concat([buggy(big, target_mode=target, lag_ms=0, decoder="kalman")
               for target in ("relative_velocity", "relative_position")], ignore_index=True)
cat_panel(ax[0, 2], s, "target_mode", "M2_mean", "(3) target  [lag 0, kalman]  BUGGY",
          ["relative_velocity", "relative_position"])

# 4-6 identical to the fixed figure (dimension_sweep, unaffected by the bug)
s = dim[(dim.trial_mode == "single_trial") & (dim.d == 2)]
line_panel(ax[1, 0], s, "K_PCS", "decode", "(4) PCA dim  [velocity]  (unaffected)", "K_PCS")
s = dim[(dim.trial_mode == "single_trial") & (dim.K_PCS == 12)]
line_panel(ax[1, 1], s, "d", "decode", "(5) CCA dim  [K_PCS=12, velocity]  (unaffected)", "CCA dims kept")
ax[1, 1].set_xticks(sorted(s.d.unique()))
s = dim[(dim.K_PCS == 12) & (dim.d == 2)]
cat_panel(ax[1, 2], s, "trial_mode", "decode", "(6) alignment mode  [K_PCS=12, velocity]  (unaffected)",
          ["average", "single_trial"])

from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([0], [0], color=COL["R2->R1"], marker="o", lw=2, label="R2→R1"),
    Line2D([0], [0], color=COL["R1->R2"], marker="o", lw=2, label="R1→R2"),
    Line2D([0], [0], color=COL["R1->R1"], marker="o", lw=2, label="R1→R1 (panels 4–6)"),
], loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.955))
fig.suptitle("UNFIXED / BUGGY: exclude-only filter → cross-epoch panels (1–3) represent 0828 ONLY "
             "(label says n(R2)=3, but 0829/0830 silently dropped)", fontsize=12.5, y=1.0, color="#b00")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
# report how many pairs each buggy cross-epoch panel actually used
s2 = buggy(big, target_mode="relative_position", lag_ms=0, decoder="kalman")
print("buggy decoder-panel kalman cross-epoch pairs:",
      s2[s2.pair.isin(["R1->R2", "R2->R1"])].groupby("pair").size().to_dict())
