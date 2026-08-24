"""Focused analysis: train decoder on each r1 day, test on 0829 (the one clean r2 day).

This pulls from the already-computed 16x16 cross-day matrix and asks one
specific question: does the cross-epoch drop (r1_train -> 0829_test) exceed
what we see for r1_train -> other_r1_day transfers?

Per train day we compare three numbers:
  - within_day        : 5-fold CV within the train day (upper bound)
  - mean_to_other_r1  : average corr from this train day to the other 12 r1 days
                        (the "natural drift" baseline)
  - corr_to_0829      : corr from this train day to 0829 (the cross-epoch test)

If task B left a fingerprint on the neural representation, we expect
corr_to_0829 < mean_to_other_r1 paired across days.

We report:
  - per-day table
  - paired Wilcoxon signed-rank test (mean_to_other_r1 vs corr_to_0829)
  - bar plot
  - distribution overlay (off-diag r1<->r1 vs r1->0829)

Outputs:
  Results/generalization/r1_to_0829_table.csv
  Results/generalization/r1_to_0829.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS_DIR))

REPO_ROOT = _THIS_DIR.parents[1]
RES = REPO_ROOT / "Results" / "generalization"

TARGET_DATE = "20250829"


def main():
    long_df = pd.read_csv(RES / "cross_day_corr_long.csv")
    # Primary metric is M2 (per-trial Pearson r). Fall back to the legacy
    # per-dim concat corr if the new column isn't there yet.
    metric_col = "corr_per_trial_mean" if "corr_per_trial_mean" in long_df.columns else "corr"
    per_pair = (
        long_df.groupby(["train_date", "test_date", "train_epoch", "test_epoch"])[metric_col]
        .mean().reset_index(name="corr")
    )

    r1_dates = sorted(per_pair.loc[per_pair.train_epoch == "r1", "train_date"].unique())

    rows = []
    for d in r1_dates:
        within = per_pair[(per_pair.train_date == d) & (per_pair.test_date == d)]["corr"]
        other_r1 = per_pair[
            (per_pair.train_date == d)
            & (per_pair.test_epoch == "r1")
            & (per_pair.test_date != d)
        ]["corr"]
        to_0829 = per_pair[
            (per_pair.train_date == d) & (per_pair.test_date == int(TARGET_DATE))
        ]["corr"]
        rows.append({
            "train_date": d,
            "within_day": float(within.iloc[0]) if len(within) else np.nan,
            "mean_to_other_r1": float(other_r1.mean()) if len(other_r1) else np.nan,
            "corr_to_0829": float(to_0829.iloc[0]) if len(to_0829) else np.nan,
        })
    tbl = pd.DataFrame(rows)
    tbl["diff_r1_minus_0829"] = tbl["mean_to_other_r1"] - tbl["corr_to_0829"]
    tbl.to_csv(RES / "r1_to_0829_table.csv", index=False)

    print("=== Per-train-day comparison ===")
    print(tbl.round(3).to_string(index=False))

    # paired test
    a = tbl["mean_to_other_r1"].to_numpy()
    b = tbl["corr_to_0829"].to_numpy()
    diff = a - b
    print()
    print(f"n train days = {len(tbl)}")
    print(f"mean(other_r1) = {np.mean(a):.3f}  mean(to_0829) = {np.mean(b):.3f}")
    print(f"mean paired diff = {np.mean(diff):+.3f}  median = {np.median(diff):+.3f}")
    w_stat, w_p = stats.wilcoxon(a, b, alternative="greater")
    print(f"Wilcoxon signed-rank (H1: other_r1 > to_0829): W={w_stat:.1f}, p={w_p:.4g}")
    # also one-sided t-test for backup
    t_stat, t_p = stats.ttest_rel(a, b, alternative="greater")
    print(f"Paired t-test (one-sided):                   t={t_stat:.2f}, p={t_p:.4g}")

    # ===== Figure =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    x = np.arange(len(tbl))
    w = 0.28
    ax1.bar(x - w, tbl["within_day"], w, label="within day (CV)", color="tab:blue")
    ax1.bar(x, tbl["mean_to_other_r1"], w, label="→ other r1 (mean)", color="tab:green")
    ax1.bar(x + w, tbl["corr_to_0829"], w, label=f"→ {TARGET_DATE}", color="tab:red")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tbl["train_date"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Mean velocity correlation")
    ax1.set_title("Per r1 training day: where can the decoder generalize?")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3, axis="y")

    # distributions
    off_r1_r1 = per_pair[
        (per_pair.train_epoch == "r1")
        & (per_pair.test_epoch == "r1")
        & (per_pair.train_date != per_pair.test_date)
    ]["corr"].to_numpy()
    r1_to_0829 = per_pair[
        (per_pair.train_epoch == "r1")
        & (per_pair.test_date == int(TARGET_DATE))
    ]["corr"].to_numpy()
    bins = np.linspace(-0.05, 0.4, 25)
    ax2.hist(off_r1_r1, bins=bins, alpha=0.6, color="tab:green",
             label=f"r1 → other r1 (n={len(off_r1_r1)})\nmean={off_r1_r1.mean():.3f}")
    ax2.hist(r1_to_0829, bins=bins, alpha=0.7, color="tab:red",
             label=f"r1 → {TARGET_DATE} (n={len(r1_to_0829)})\nmean={r1_to_0829.mean():.3f}")
    ax2.axvline(off_r1_r1.mean(), color="tab:green", linestyle="--", linewidth=1)
    ax2.axvline(r1_to_0829.mean(), color="tab:red", linestyle="--", linewidth=1)
    ax2.set_xlabel("Mean velocity correlation")
    ax2.set_ylabel("Number of (train, test) pairs")
    ax2.set_title(f"Distributions  (paired Wilcoxon p = {w_p:.3g})")
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = RES / "r1_to_0829.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
