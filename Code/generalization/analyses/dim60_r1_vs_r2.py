"""H2 slide panel: neural dimensionality of R1 vs R2 — PR and #PCs for 60% variance.

Left panel of fig_subspace_inclusion, recomputed at a 60% variance threshold (the stored
CSV only has dim80/dim90). Same loading config as analyses/subspace_inclusion.py.
Lower PR / lower dim60 in R2 = R2 is the lower-dimensional (more compressed) epoch.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from subspace_inclusion import load  # same NWB/bin/smoothing config
from big_sweep_phase2_crossday import SESSIONS_R1, SESSIONS_R2, EXCLUDE_TRIALS

REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "dim60_r1_vs_r2.csv"
FIG = REPO / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_dim60_r1_vs_r2.png"
VAR_TARGET = 0.60


def spectrum(M, target=VAR_TARGET):
    """Participation ratio and #PCs reaching `target` cumulative variance."""
    Mc = M - M.mean(0)
    lam = np.linalg.eigvalsh(np.cov(Mc.T))
    lam = np.clip(lam[::-1], 0, None)
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    cum = np.cumsum(lam) / lam.sum()
    return pr, int(np.argmax(cum >= target) + 1)


rows = []
for s in SESSIONS_R1 + SESSIONS_R2:
    _, _, _, Ysm = load(s, EXCLUDE_TRIALS.get(s, []))
    pr, d60 = spectrum(Ysm)
    rows.append(dict(session=s.replace("TSAL", "")[:8],
                     epoch="R2" if s in SESSIONS_R2 else "R1", PR=pr, dim60=d60))
A = pd.DataFrame(rows)
A.to_csv(OUT_CSV, index=False)
print(A.groupby("epoch")[["PR", "dim60"]].mean().round(2).to_string())

C = {"R1": "#8d9295", "R2": "#e0553f"}
fig, ax = plt.subplots(figsize=(8, 5.6))
rng = np.random.default_rng(0)
# points occupy the LEFT half of each slot, mean line + label the RIGHT half -> no occlusion
spec = [(0, "R1", "PR"), (1.15, "R2", "PR"), (2.9, "R1", "dim60"), (4.05, "R2", "dim60")]
for x, ep, col in spec:
    v = A[A.epoch == ep][col].to_numpy(float)
    ax.scatter(np.full(len(v), x - 0.20) + rng.normal(0, .05, len(v)), v,
               color=C[ep], s=70, alpha=.75, edgecolors="w", lw=.8, zorder=2)
    ax.hlines(v.mean(), x + 0.04, x + 0.42, color=C[ep], lw=3.5, zorder=4)
    ax.text(x + 0.47, v.mean(), f"{v.mean():.1f}", va="center", ha="left", fontsize=12,
            color=C[ep], fontweight="bold", zorder=5)
ax.axvline(2.02, color="#ccc", lw=1, ls="--")
ax.set_xlim(-0.6, 4.8)
ax.set_xticks([p[0] for p in spec])
ax.set_xticklabels([f"{ep}\n{col}" for _, ep, col in spec], fontsize=11)
ax.set_ylabel("value", fontsize=11)
ax.set_title("Neural dimensionality in each session's own space\n"
             "R2 is lower-dimensional than R1 (PR and #PCs for 60 % variance)", fontsize=12)
ax.grid(alpha=.25, axis="y")
ax.text(0.5, -0.16, f"one animal · n(R1)={sum(A.epoch=='R1')}, n(R2)={sum(A.epoch=='R2')} · "
                    "single-trial population spectrum",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#888")
fig.tight_layout()
fig.savefig(FIG, dpi=180, bbox_inches="tight")
print("saved", FIG)
