"""Honest forward-vs-reverse bar for the 'Asymmetrical generalization' slide.

Two bars (R1->R2 forward, R2->R1 reverse), position, using the OUT-OF-SAMPLE nested CCA
(trial-average, d=12) per-R2-day values. Overlays the three R2-day points and connects each
day's forward/reverse pair (dumbbell) so the per-day spread is visible, not hidden in a mean.
All three R2 days have reverse > forward (asymmetry +0.206 / +0.185 / +0.095).
"""
from __future__ import annotations
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "nested_cca_by_r2_session.csv"
OUT = REPO / "Results" / "workflows" / "generalization" / "figures" / "fig_asymmetry_bar_nested_position.png"

d = pd.read_csv(CSV)
s = d[(d.target == "relative_position") & (d.alignment_mode == "average")
      & (d.method == "nested") & (d.dims == 12)].sort_values("r2_session")
fwd = s.forward_corr.to_numpy()
rev = s.reverse_corr.to_numpy()
days = [re.search(r"(\d{8})", str(x)).group(1)[-4:] for x in s.r2_session]   # 0828/0829/0830
asym = rev - fwd

FCOL, RCOL = "#e74c3c", "#3498db"   # forward red, reverse blue
fig, ax = plt.subplots(figsize=(5.2, 6))

ax.bar(0, fwd.mean(), width=0.55, color=FCOL, alpha=0.28, edgecolor=FCOL, lw=2, zorder=1)
ax.bar(1, rev.mean(), width=0.55, color=RCOL, alpha=0.28, edgecolor=RCOL, lw=2, zorder=1)

rng = np.random.default_rng(0)
jit = np.linspace(-0.12, 0.12, len(days))
for i, day in enumerate(days):
    ax.plot([0 + jit[i], 1 + jit[i]], [fwd[i], rev[i]], "-", color="#888", lw=1.4, zorder=2)
    ax.scatter(0 + jit[i], fwd[i], s=70, color=FCOL, edgecolor="k", zorder=3)
    ax.scatter(1 + jit[i], rev[i], s=70, color=RCOL, edgecolor="k", zorder=3)
    ax.annotate(day, (1 + jit[i], rev[i]), textcoords="offset points", xytext=(9, 0),
                fontsize=8, color="#555", va="center")

# mean labels
ax.text(0, fwd.mean() - 0.03, f"{fwd.mean():.2f}", ha="center", va="top", fontsize=11, color=FCOL, fontweight="bold")
ax.text(1, rev.mean() + 0.015, f"{rev.mean():.2f}", ha="center", va="bottom", fontsize=11, color=RCOL, fontweight="bold")

ax.set_xticks([0, 1])
ax.set_xticklabels(["R1→R2\n(forward)", "R2→R1\n(reverse)"], fontsize=12)
ax.set_ylabel("cross-day decode corr (position)", fontsize=11)
ax.set_ylim(0, max(rev.max(), fwd.max()) * 1.18)
ax.set_title("Reverse > forward on all 3 R2 days\n"
             f"asymmetry +{asym[0]:.2f} / +{asym[1]:.2f} / +{asym[2]:.2f}  ·  nested (out-of-sample) CCA",
             fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.5, -0.14, "position · trial-average nested CCA · d=12 · one animal, n(R2)=3",
        transform=ax.transAxes, ha="center", fontsize=8, color="#888")
fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print("forward per day:", [round(x, 3) for x in fwd])
print("reverse per day:", [round(x, 3) for x in rev])
print("asymmetry     :", [round(x, 3) for x in asym])
print("saved", OUT)
