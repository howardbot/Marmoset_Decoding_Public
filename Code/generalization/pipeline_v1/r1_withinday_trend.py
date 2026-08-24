"""Plot per-session within-day decoder correlation across the 13 r1 sessions
at the locked configuration (20 ms bin, sigma = 50 ms causal Gaussian,
velocity target, k = 15 PCs, per-session optimal lag).

Reads the diagonal entries of the cross-day matrix from
Results/generalization/cross_day_corr_long.csv (which already contains
M1 = corr_concat_mean and M2 = corr_per_trial_mean).

Outputs:
  Results/legacy/report_figures/r1_withinday_trend.png
  Results/generalization/r1_withinday_trend.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parents[1]
RES = _THIS_DIR.parents[1] / "Results" / "generalization"
FIG_DIR = _THIS_DIR.parents[1] / "Results" / "legacy" / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    long_df = pd.read_csv(RES / "cross_day_corr_long.csv")
    # r1 diagonal entries (within-day)
    r1 = long_df[(long_df.train_epoch == "r1") & (long_df.test_epoch == "r1")]
    diag = r1[r1.train_date == r1.test_date]
    # collapse across the 3 vel dims per (train, test) cell — these columns are
    # already constant per pair, so .first() is fine
    per_sess = diag.groupby("train_date").agg(
        M1=("corr_concat_mean", "first"),
        M2=("corr_per_trial_mean", "first"),
    ).reset_index().sort_values("train_date")

    print("=== Per-session within-day decoder corr (locked config) ===")
    print(per_sess.round(3).to_string(index=False))
    print(f"\nM1 mean = {per_sess.M1.mean():.3f}  (range {per_sess.M1.min():.3f}–{per_sess.M1.max():.3f})")
    print(f"M2 mean = {per_sess.M2.mean():.3f}  (range {per_sess.M2.min():.3f}–{per_sess.M2.max():.3f})")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(per_sess))
    w = 0.38

    ax.bar(x - w / 2, per_sess.M1, w, label="M1 (concat Pearson r)",
           color="tab:orange", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, per_sess.M2, w, label="M2 (per-trial Pearson r)",
           color="tab:blue", edgecolor="white", linewidth=0.5)

    # mean reference lines
    ax.axhline(per_sess.M1.mean(), color="tab:orange", linestyle="--", linewidth=1,
               label=f"M1 mean = {per_sess.M1.mean():.3f}")
    ax.axhline(per_sess.M2.mean(), color="tab:blue", linestyle="--", linewidth=1,
               label=f"M2 mean = {per_sess.M2.mean():.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(per_sess.train_date.astype(int).astype(str), rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Session date")
    ax.set_ylabel("Within-day decoder correlation (5-fold CV in PC space)")
    ax.set_title("R1 per-session within-day decoder corr — locked config "
                 "(20 ms bin, σ=50 ms, vel, k=15)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(per_sess.M2.max(), per_sess.M1.max()) + 0.08)

    fig.tight_layout()
    out_results = RES / "r1_withinday_trend.png"
    out_figs = FIG_DIR / "r1_withinday_trend.png"
    fig.savefig(out_results, dpi=150)
    fig.savefig(out_figs, dpi=150)
    print(f"\nSaved {out_results}")
    print(f"Saved {out_figs}")


if __name__ == "__main__":
    main()
