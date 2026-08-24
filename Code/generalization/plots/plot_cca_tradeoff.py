"""CCA dimensionality trade-off, anchored on the within-epoch R1->R1 cross-day reference.

Two things the review asked for and v1 did not show:
  1. held-out CC split BY pair category, with R1->R1 (different sessions of the
     same epoch) drawn as the positive-control reference. The leading three dims are
     robustly shared cross-epoch; dim 4 is marginal/category-dependent.
  2. R1->R1 decode drawn as a reference on the right axis, so R1->R2 and R2->R1
     can be compared with within-epoch cross-day decoding.

Left axis (purple): per-dim held-out CC, R1->R1 (solid reference) vs R1->R2 (dashed).
Right axis: cross-day decode corr vs #CCA dims, R1->R1 / R1->R2 / R2->R1.
Reads Results/workflows/generalization/cca_sweep_long.csv ; writes fig_cca_tradeoff.png
"""
from __future__ import annotations
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
c = pd.read_csv(REPO / "Results" / "workflows" / "generalization" / "cca_sweep_long.csv")
OUT = REPO / "Results" / "workflows" / "generalization" / "figures" / "fig_cca_tradeoff.png"


def pick(df):
    """Prefer the 0828-excluded rows; R1->R1 pairs never involve 0828 so they only
    exist as 'include' — fall back to that."""
    exc = df[df.outlier_mode == "exclude"]
    return exc if len(exc) else df[df.outlier_mode == "include"]


def by_dim(df, metric, pair_cat, target=None):
    g = df[(df.metric == metric) & (df.pair_category == pair_cat)]
    if target is not None:
        g = g[g.target_mode == target]
    g = pick(g)
    return g.groupby("n_cca")["corr"].mean().sort_index()


fig, ax = plt.subplots(figsize=(9.5, 5.5))

# --- Left axis: held-out CC per-dim, within-epoch reference vs cross-epoch ---
cc_r1r1 = by_dim(c, "cca", "R1->R1")
cc_r1r2 = by_dim(c, "cca", "R1->R2")
ax.plot(cc_r1r1.index, cc_r1r1.values, color="#8e44ad", lw=2.5, marker="o", ms=4,
        label="held-out CC · R1→R1 (within-epoch cross-day)")
ax.plot(cc_r1r2.index, cc_r1r2.values, color="#8e44ad", lw=1.4, ls=":", marker="s", ms=3,
        alpha=.75, label="held-out CC · R1→R2 (cross-day)")
ax.set_xlabel("# CCA dims kept")
ax.set_ylabel("held-out canonical correlation", color="#8e44ad")
ax.set_ylim(0, 1.02)
ax.set_xticks(sorted(c[c.metric == "cca"].n_cca.unique()))
ax.axhline(0.2, color="#8e44ad", ls="--", lw=0.8, alpha=.5)
ax.text(4.3, 0.80,
        "three leading dims robustly shared cross-epoch\n"
        "dim 4 is marginal/category-dependent; later dims\n"
        "approach the finite-sample floor",
        fontsize=8.5, color="#5b2c6f")

# --- Right axis: decode corr, R1->R1 reference + forward/reverse ---
ax2 = ax.twinx()
for pc, col, lab in [("R1->R1", "#7f8c8d", "decode R1→R1 (reference)"),
                     ("R2->R1", "#3498db", "decode R2→R1 (reverse)"),
                     ("R1->R2", "#e74c3c", "decode R1→R2 (forward)")]:
    g = by_dim(c, "decode", pc, target="relative_velocity")
    zo = 1 if pc == "R1->R1" else 3
    ax2.plot(g.index, g.values, "--o", color=col, ms=5, lw=1.8, zorder=zo, label=lab)
ax2.set_ylabel("cross-day decode corr (velocity)")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8, framealpha=.9)
ax.set_title("CCA dimensionality trade-off (K_PCS=12), anchored on R1→R1\n"
             "held-out CC (purple): three robust leading shared dims; "
             "high-d decode differences include weakly aligned components",
             fontsize=10)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
