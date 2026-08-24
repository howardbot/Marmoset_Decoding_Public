"""Plot the cross-day decoder transfer matrix and a within-vs-cross summary.

Two panels side by side:
  Left  : 13x13 heatmap of mean velocity correlation, diagonal entries outlined
          in red so the within-day baseline is visually separable.
  Right : Per-train-day bar chart of within-day CV correlation vs the mean
          off-diagonal correlation. The gap is the "generalization drop."

Reads `Results/generalization/cross_day_corr_matrix.csv` and
`Results/generalization/generalization_summary.csv`, writes the figure to
`Results/generalization/cross_day_corr_matrix.png`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "Results" / "generalization"


R1_LAST_DATE = 20250813
R2_FIRST_DATE = 20250828

# Find R2 index in the matrix
def _epoch_split_index(date_list):
    """Return the index just before the first r2 entry, or None if no r2 present."""
    for i, d in enumerate(date_list):
        if int(d) >= R2_FIRST_DATE:
            return i
    return None


def main():
    # Primary metric is now M2 (per-trial Pearson r), but we also load M1 for
    # the back-compat panel.
    mat_path = RESULTS_DIR / "cross_day_corr_per_trial_matrix.csv"
    if not mat_path.exists():
        # fall back to the legacy single-metric matrix
        mat_path = RESULTS_DIR / "cross_day_corr_matrix.csv"
    mat = pd.read_csv(mat_path, index_col=0)
    summary = pd.read_csv(RESULTS_DIR / "generalization_summary.csv")

    null_long_path = RESULTS_DIR / "cross_day_null_long.csv"
    long_path = RESULTS_DIR / "cross_day_corr_long.csv"
    has_null = null_long_path.exists() and long_path.exists()

    if has_null:
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 0.9]}
        )
    else:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.3, 1]}
        )

    vmax = max(0.4, float(np.nanmax(mat.values)))
    im = ax1.imshow(mat.values, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
    ax1.set_xticks(range(len(mat.columns)))
    ax1.set_yticks(range(len(mat.index)))
    ax1.set_xticklabels([str(c) for c in mat.columns], rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels([str(c) for c in mat.index], fontsize=8)
    ax1.set_xlabel("Test day")
    ax1.set_ylabel("Train day")
    ax1.set_title("Cross-day decoder transfer (manifold-aligned, M2 per-trial corr)")
    plt.colorbar(im, ax=ax1, label="per-trial Pearson r (averaged)")
    # Highlight diagonal so the within-day CV baseline is visually distinct
    for i in range(len(mat)):
        ax1.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=1.5,
            )
        )
    # Mark the r1 / r2 epoch boundary if both epochs are present
    split_x = _epoch_split_index(list(mat.columns))
    split_y = _epoch_split_index(list(mat.index))
    if split_x is not None:
        ax1.axvline(split_x - 0.5, color="white", linewidth=2.0)
    if split_y is not None:
        ax1.axhline(split_y - 0.5, color="white", linewidth=2.0)

    x = np.arange(len(summary))
    w = 0.38
    ax2.bar(x - w / 2, summary.within_day, w, label="within day (5-fold CV)", color="tab:blue")
    ax2.bar(x + w / 2, summary.mean_off_diag, w, label="mean off-diagonal", color="tab:orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in summary.train_date], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Mean velocity correlation")
    ax2.set_title("Within-day vs cross-day, per training session")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # Right-most panel (if null available): histogram of real off-diag vs null
    if has_null:
        long_df = pd.read_csv(long_path)
        null_df = pd.read_csv(null_long_path)
        # off-diag pairs only
        off_real = long_df[long_df.train_date != long_df.test_date]
        off_null = null_df[null_df.train_date != null_df.test_date]
        # primary metric is M2 if available, else fall back to the per-dim
        # concat corr already in the legacy CSV
        if "corr_per_trial_mean" in off_real.columns:
            real_vals = off_real.groupby(["train_date", "test_date"])["corr_per_trial_mean"].first().values
        else:
            real_vals = off_real.groupby(["train_date", "test_date"])["corr"].mean().values
        null_vals = off_null.groupby(["train_date", "test_date"])["null_corr"].mean().values
        bins = np.linspace(-0.1, 0.4, 41)
        ax3.hist(null_vals, bins=bins, alpha=0.55, color="grey", label=f"null (circ-shift)\nmean={null_vals.mean():+.3f}")
        ax3.hist(real_vals, bins=bins, alpha=0.7, color="tab:orange", label=f"real off-diag\nmean={real_vals.mean():+.3f}")
        ax3.axvline(0, color="k", linewidth=0.8, linestyle=":")
        ax3.set_xlabel("Mean velocity correlation")
        ax3.set_ylabel("Number of (train, test) pairs")
        ax3.set_title("Null vs real off-diagonal distribution")
        ax3.legend(loc="best", fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = RESULTS_DIR / "cross_day_corr_matrix.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
