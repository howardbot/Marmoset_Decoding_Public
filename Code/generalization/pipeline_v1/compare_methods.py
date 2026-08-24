"""Side-by-side comparison: Origin Kording concat metric (M1) vs per-trial metric (M2)
on the r1 internal cross-day matrix.

Same Kalman predictions, same data, two different evaluation conventions:
  Origin Kording style (M1) = concatenate all test bins -> one Pearson r per dim,
                              then average across the 3 velocity dims.
  Per-trial (M2)            = compute Pearson r within each test trial per dim,
                              then average across trials and dims.

Outputs:
  Results/generalization/methods_heatmap.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
RES = REPO_ROOT / "Results" / "generalization"


def main():
    long_df = pd.read_csv(RES / "cross_day_corr_long.csv")
    r1 = long_df[(long_df.train_epoch == "r1") & (long_df.test_epoch == "r1")]

    pair = r1.groupby(["train_date", "test_date"]).first().reset_index()
    M1 = pair.pivot(index="train_date", columns="test_date", values="corr_concat_mean")
    M2 = pair.pivot(index="train_date", columns="test_date", values="corr_per_trial_mean")

    # ---- numeric summary ----
    diag1 = np.diag(M1.values)
    diag2 = np.diag(M2.values)
    off1 = M1.values[~np.eye(len(M1), dtype=bool)]
    off2 = M2.values[~np.eye(len(M2), dtype=bool)]
    r_pair, p_pair = stats.pearsonr(off1, off2)

    print("=== Diagonal (within-day, 5-fold CV in PC space) ===")
    print(f"  M1 (Origin concat):  mean={diag1.mean():+.3f}  range=[{diag1.min():+.3f}, {diag1.max():+.3f}]")
    print(f"  M2 (per-trial):      mean={diag2.mean():+.3f}  range=[{diag2.min():+.3f}, {diag2.max():+.3f}]")
    print()
    print("=== Off-diagonal (cross-day, n=156 pairs) ===")
    print(f"  M1 (Origin concat):  mean={off1.mean():+.3f}  range=[{off1.min():+.3f}, {off1.max():+.3f}]")
    print(f"  M2 (per-trial):      mean={off2.mean():+.3f}  range=[{off2.min():+.3f}, {off2.max():+.3f}]")
    print()
    print(f"  pair-level Pearson r(M1, M2) = {r_pair:+.3f}, p = {p_pair:.3g}")
    print(f"  paired difference M2 − M1   = {(off2 - off1).mean():+.3f}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 5),
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    vmin = 0.0
    vmax = max(M1.values.max(), M2.values.max())

    for ax, mat, title in [
        (axes[0], M1, "Origin Kording (concat Pearson r)"),
        (axes[1], M2, "Per-trial (Pearson r averaged across trials)"),
    ]:
        im = ax.imshow(mat.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_yticks(range(len(mat.index)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_xlabel("Test day")
        ax.set_ylabel("Train day")
        ax.set_title(title)
        for i in range(len(mat)):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=1.4))
        plt.colorbar(im, ax=ax, label="mean velocity correlation")

    # Right: per-pair scatter
    ax = axes[2]
    ax.scatter(off1, off2, alpha=0.55, s=22, color="tab:purple")
    lo = min(off1.min(), off2.min())
    hi = max(off1.max(), off2.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("M1 Origin concat corr")
    ax.set_ylabel("M2 per-trial corr")
    ax.set_title(f"Per-pair comparison (n=156)\nr = {r_pair:+.2f},  M2 − M1 = {(off2 - off1).mean():+.3f}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Task A R1 — same Kalman, two evaluation methods")
    fig.tight_layout()
    out = RES / "methods_heatmap.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
