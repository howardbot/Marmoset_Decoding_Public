"""Slide panel: single-unit tuning quality (cvR2) + position modulation, R1 vs R2.
Subplots 1 & 3 of the tuning-distributions figure. Repertoire-level (units not tracked
across days), n(R2)=3. Shows R1 ~ R2 -> the single-unit repertoire is preserved.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CSV = REPO / "Results" / "manifold_geometry" / "tuning_distributions.csv"
OUT = REPO / "Results" / "generalization" / "figures" / "fig_tuning_1and3.png"

d = pd.read_csv(CSV)
R1C, R2C = "#8d9295", "#e07a6b"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
specs = [("cvR2", "cross-validated R²  (tuning quality)", "tuning quality (cvR2)"),
         ("mod_pos", "position modulation depth", "position modulation")]
for ax, (col, xlab, title) in zip(axes, specs):
    lo, hi = np.nanpercentile(d[col], [0.5, 99.5])
    bins = np.linspace(lo, hi, 34)
    for ep, c in [("R1", R1C), ("R2", R2C)]:
        v = d[d.epoch == ep][col].dropna()
        ax.hist(v, bins=bins, density=True, alpha=0.55, color=c, label=f"{ep} (med {v.median():.3f})")
    ax.set_xlabel(xlab); ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
axes[0].set_ylabel("probability density (per-unit)")
fig.suptitle("Single-unit tuning: R1 ≈ R2 — preserved repertoire (consistent with reassociation)\n"
             "repertoire-level, units not tracked across days · n(R2)=3",
             fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print("R1 vs R2 medians:  cvR2 %.3f/%.3f   mod_pos %.3f/%.3f" % (
    d[d.epoch=="R1"].cvR2.median(), d[d.epoch=="R2"].cvR2.median(),
    d[d.epoch=="R1"].mod_pos.median(), d[d.epoch=="R2"].mod_pos.median()))
print("saved", OUT)
