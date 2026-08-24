"""Four-panel deep-dive on the r1 internal 13x13 cross-day matrix.

Q1: Does cross-day decoder transfer depend on calendar day-gap?
    A regression of corr ~ |train_date - test_date| (in days). If slope is
    significantly negative, future r1<->r2 comparisons must correct for the
    extra 15-day gap to 20250828 / 0829.

Q2: Is the matrix symmetric?
    Scatter mat(i,j) vs mat(j,i) with y=x reference. Quantify with Pearson r
    between upper and lower triangle and with the mean |diff|.

Q3: Is "trainability" (row mean) correlated with "testability" (col mean)?
    Per day, compute (row_off_diag_mean, col_off_diag_mean). If correlated,
    "good sessions" are double-good; if uncorrelated, training quality vs
    test-day signal are independent.

Q4: Does manifold alignment quality predict decoder transfer?
    For each ordered (train, test) r1 pair, compute the mean of the top-5
    canonical correlations between the two days' PC trajectories. Scatter
    against the actual decoder cross-day corr. Strong correlation = manifold
    geometry is the bottleneck; weak correlation = decoder uses something
    beyond what canonical alignment captures.

Outputs:
  Results/workflows/generalization/r1_deep_dive.png
  Results/workflows/generalization/r1_canonical_corr_matrix.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR))

import decoder_utils as du  # noqa: E402
from cross_day_decoder import (  # noqa: E402
    build_session_cache_entry, list_sessions, session_date, session_epoch,
)
from manifold_align import heldout_canonical_correlations  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
RES = REPO_ROOT / "Results" / "workflows" / "generalization"

TOP_K_CANONICAL = 5  # average of top-k canonical correlations used as alignment quality


def date_to_ts(d):
    """YYYYMMDD int/str -> pandas Timestamp."""
    return pd.to_datetime(str(int(d)), format="%Y%m%d")


def main():
    long_df = pd.read_csv(RES / "cross_day_corr_long.csv")
    r1_pairs = long_df[(long_df.train_epoch == "r1") & (long_df.test_epoch == "r1")]
    metric_col = "corr_per_trial_mean" if "corr_per_trial_mean" in r1_pairs.columns else "corr"
    per_pair = (
        r1_pairs.groupby(["train_date", "test_date"])[metric_col].mean().reset_index(name="corr")
    )
    dates = sorted(per_pair.train_date.unique())
    mat = per_pair.pivot(index="train_date", columns="test_date", values="corr")
    print(f"r1 has {len(dates)} sessions, {len(per_pair)} (train, test) cells")

    # ---- Build manifold-alignment-quality matrix (Q4) ----
    print("\nBuilding cache for canonical-correlation computation ...")
    cache = {}
    for tag in [t for t in list_sessions() if session_epoch(t) == "r1"]:
        cache[session_date(tag)] = build_session_cache_entry(tag)
    print(f"Cache built for {len(cache)} sessions")

    canon_mat = pd.DataFrame(index=dates, columns=dates, dtype=float)
    for i, d_tr in enumerate(dates):
        for j, d_te in enumerate(dates):
            if i == j:
                canon_mat.loc[d_tr, d_te] = 1.0
                continue
            cc = heldout_canonical_correlations(
                cache[str(d_tr)], cache[str(d_te)]
            )
            canon_mat.loc[d_tr, d_te] = float(np.nanmean(cc[:TOP_K_CANONICAL]))
    canon_mat.to_csv(RES / "r1_canonical_corr_matrix.csv")
    print(f"Saved {RES / 'r1_canonical_corr_matrix.csv'}")

    # ---- Q1: day-gap regression ----
    pairs = per_pair[per_pair.train_date != per_pair.test_date].copy()
    pairs["gap_days"] = pairs.apply(
        lambda r: abs((date_to_ts(r.train_date) - date_to_ts(r.test_date)).days),
        axis=1,
    )
    slope, intercept, r_q1, p_q1, _ = stats.linregress(pairs["gap_days"], pairs["corr"])

    # ---- Q2: symmetry ----
    upper = []
    lower = []
    sym_diffs = []
    for i, d_a in enumerate(dates):
        for j, d_b in enumerate(dates):
            if i >= j:
                continue
            a = mat.loc[d_a, d_b]
            b = mat.loc[d_b, d_a]
            upper.append(a)
            lower.append(b)
            sym_diffs.append(a - b)
    upper = np.array(upper)
    lower = np.array(lower)
    sym_diffs = np.array(sym_diffs)
    r_sym, _ = stats.pearsonr(upper, lower)

    # ---- Q3: trainability vs testability ----
    rows = []
    for d in dates:
        row_off = mat.loc[d, [x for x in dates if x != d]].mean()
        col_off = mat.loc[[x for x in dates if x != d], d].mean()
        rows.append({
            "date": d,
            "train_mean_corr": float(row_off),
            "test_mean_corr": float(col_off),
            "within_day": float(mat.loc[d, d]),
        })
    tt_df = pd.DataFrame(rows)
    r_q3, p_q3 = stats.pearsonr(tt_df["train_mean_corr"], tt_df["test_mean_corr"])

    # ---- Q4: canonical alignment vs decoder corr ----
    q4_records = []
    for d_tr in dates:
        for d_te in dates:
            if d_tr == d_te:
                continue
            q4_records.append({
                "train_date": d_tr,
                "test_date": d_te,
                "canonical_top5": float(canon_mat.loc[d_tr, d_te]),
                "decoder_corr": float(mat.loc[d_tr, d_te]),
            })
    q4 = pd.DataFrame(q4_records)
    r_q4, p_q4 = stats.pearsonr(q4["canonical_top5"], q4["decoder_corr"])

    # ===== Plot 4 panels =====
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Q1
    ax = axes[0, 0]
    ax.scatter(pairs["gap_days"], pairs["corr"], alpha=0.5, color="tab:blue", s=18)
    xx = np.linspace(pairs["gap_days"].min(), pairs["gap_days"].max(), 50)
    ax.plot(xx, slope * xx + intercept, "k--",
            label=f"slope = {slope:+.4f}/day  r = {r_q1:+.2f}  p = {p_q1:.3g}")
    ax.set_xlabel("|train_date − test_date|  (days)")
    ax.set_ylabel("Cross-day decoder correlation")
    ax.set_title("Q1: Does generalization depend on day-gap?")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Q2
    ax = axes[0, 1]
    ax.scatter(upper, lower, alpha=0.6, color="tab:purple", s=24)
    lo, hi = min(upper.min(), lower.min()), max(upper.max(), lower.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("corr(train_i → test_j)")
    ax.set_ylabel("corr(train_j → test_i)")
    ax.set_title(
        f"Q2: Symmetry?  Pearson r(i→j, j→i) = {r_sym:+.2f}\n"
        f"mean |asymmetry| = {np.mean(np.abs(sym_diffs)):.3f}"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Q3
    ax = axes[1, 0]
    ax.scatter(tt_df["train_mean_corr"], tt_df["test_mean_corr"],
               c=tt_df["within_day"], cmap="viridis", s=60, edgecolor="k")
    for _, r in tt_df.iterrows():
        ax.annotate(
            str(int(r["date"]))[-4:],
            (r["train_mean_corr"], r["test_mean_corr"]),
            fontsize=7, xytext=(3, 3), textcoords="offset points",
        )
    lo, hi = min(tt_df["train_mean_corr"].min(), tt_df["test_mean_corr"].min()) - 0.01, \
             max(tt_df["train_mean_corr"].max(), tt_df["test_mean_corr"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x")
    ax.set_xlabel("Trainability (row mean off-diag)")
    ax.set_ylabel("Testability (col mean off-diag)")
    ax.set_title(f"Q3: trainability vs testability   Pearson r = {r_q3:+.2f}  p = {p_q3:.3g}")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=tt_df["within_day"].min(),
                                                  vmax=tt_df["within_day"].max()))
    fig.colorbar(sm, ax=ax, label="within-day CV corr")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Q4
    ax = axes[1, 1]
    ax.scatter(q4["canonical_top5"], q4["decoder_corr"], alpha=0.5, color="tab:green", s=18)
    slope4, intercept4, _, _, _ = stats.linregress(q4["canonical_top5"], q4["decoder_corr"])
    xx = np.linspace(q4["canonical_top5"].min(), q4["canonical_top5"].max(), 50)
    ax.plot(xx, slope4 * xx + intercept4, "k--",
            label=f"slope = {slope4:+.2f}  r = {r_q4:+.2f}  p = {p_q4:.3g}")
    ax.set_xlabel(f"Mean top-{TOP_K_CANONICAL} canonical correlation (alignment quality)")
    ax.set_ylabel("Cross-day decoder correlation")
    ax.set_title("Q4: Does manifold alignment quality predict decoder transfer?")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("r1 internal 13×13: deep dive (n=156 ordered pairs)")
    fig.tight_layout()
    out = RES / "r1_deep_dive.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")

    # ----- Print summary -----
    print("\n=== Summary ===")
    print(f"Q1: slope = {slope:+.4f} corr/day, r = {r_q1:+.2f}, p = {p_q1:.3g}")
    print(f"    -> day-gap explains {r_q1**2*100:.1f}% of variance in cross-day corr")
    print(f"Q2: r(i→j, j→i) = {r_sym:+.2f}; mean |asymmetry| = {np.mean(np.abs(sym_diffs)):.3f}")
    print(f"    -> matrix is "
          f"{'highly symmetric' if r_sym > 0.7 else 'moderately symmetric' if r_sym > 0.4 else 'weakly symmetric'}")
    print(f"Q3: trainability vs testability r = {r_q3:+.2f}, p = {p_q3:.3g}")
    print(f"Q4: canonical_top{TOP_K_CANONICAL} vs decoder corr r = {r_q4:+.2f}, p = {p_q4:.3g}")
    print(f"    -> alignment quality explains {r_q4**2*100:.1f}% of decoder-transfer variance")


if __name__ == "__main__":
    main()
