"""Q4 follow-up: which manifold-alignment metric predicts cross-day decoder transfer?

The first pass used the mean of the top-5 canonical correlations and got r ≈ 0
because every r1 pair has top-5 canonical correlations ≳ 0.99 — alignment is
basically saturated in the top subspace, so this metric has no variance to predict
anything. The decoder transfer correlation, by contrast, varies from ~0.06 to ~0.32.
That gap is what this script tries to localize.

For every ordered r1 pair we compute the full 15-component canonical correlation
profile and a battery of summary statistics:
  - mean top-k for k ∈ {3, 5, 10, 15}
  - mean of the last 5 components (the noisiest, least-aligned subspace)
  - minimum canonical correlation (worst aligned dimension)
  - "effective dim" = number of components with cc >= 0.8 (manifold "thickness")
  - tail mass = mean of cc beyond the top-5
  - sum (area under the decay curve)

We then regress each metric against the decoder transfer correlation. The metric
with the largest |r| tells us which slice of the manifold actually matters for
behavioral decoding generalization.

Outputs:
  Results/workflows/generalization/q4_canonical_profile_per_pair.csv  -- 156 pairs x 15 cc + metrics
  Results/workflows/generalization/q4_metric_correlations.csv         -- summary table
  Results/workflows/generalization/q4_followup.png                    -- 4-panel figure
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

from cross_day_decoder import (  # noqa: E402
    build_session_cache_entry, list_sessions, session_date, session_epoch, K_PCS,
)
from manifold_align import heldout_canonical_correlations  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
RES = REPO_ROOT / "Results" / "workflows" / "generalization"

EFFECTIVE_DIM_THRESHOLD = 0.8  # canonical correlation considered "well aligned"


def metric_battery(cc):
    """Compute summary metrics from a length-K canonical-correlation vector."""
    cc = np.asarray(cc, dtype=float)
    k = len(cc)
    out = {
        "mean_top3": float(np.mean(cc[:3])),
        "mean_top5": float(np.mean(cc[:5])),
        "mean_top10": float(np.mean(cc[:10])),
        "mean_all": float(np.mean(cc)),
        "mean_last5": float(np.mean(cc[-5:])),
        "mean_tail_beyond_top5": float(np.mean(cc[5:])) if k > 5 else np.nan,
        "min_cc": float(np.min(cc)),
        "effective_dim": int(np.sum(cc >= EFFECTIVE_DIM_THRESHOLD)),
        "area_under_curve": float(np.sum(cc)),
    }
    return out


def main():
    long_df = pd.read_csv(RES / "cross_day_corr_long.csv")
    r1_pairs = long_df[(long_df.train_epoch == "r1") & (long_df.test_epoch == "r1")]
    metric_col = "corr_per_trial_mean" if "corr_per_trial_mean" in r1_pairs.columns else "corr"
    per_pair = (
        r1_pairs.groupby(["train_date", "test_date"])[metric_col].mean().reset_index(name="decoder_corr")
    )
    dates = sorted(per_pair.train_date.unique())

    print(f"Building r1 PC trajectories for {len(dates)} sessions ...")
    cache = {}
    for tag in [t for t in list_sessions() if session_epoch(t) == "r1"]:
        cache[session_date(tag)] = build_session_cache_entry(tag)

    # Per pair: full canonical correlation profile + metrics
    rows = []
    for d_tr in dates:
        for d_te in dates:
            if d_tr == d_te:
                continue
            cc = heldout_canonical_correlations(cache[str(d_tr)], cache[str(d_te)])
            metrics = metric_battery(cc)
            row = {"train_date": d_tr, "test_date": d_te}
            row.update({f"cc_{i+1}": float(cc[i]) for i in range(len(cc))})
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.merge(per_pair, on=["train_date", "test_date"], how="left")
    df.to_csv(RES / "q4_canonical_profile_per_pair.csv", index=False)
    print(f"Saved {RES / 'q4_canonical_profile_per_pair.csv'}  ({len(df)} pairs)")

    # ---- Test each metric against decoder corr ----
    metric_names = [
        "mean_top3", "mean_top5", "mean_top10", "mean_all",
        "mean_last5", "mean_tail_beyond_top5",
        "min_cc", "effective_dim", "area_under_curve",
    ]
    res = []
    for m in metric_names:
        r, p = stats.pearsonr(df[m], df["decoder_corr"])
        res.append({"metric": m, "pearson_r": r, "p_value": p,
                    "spread_min": float(df[m].min()), "spread_max": float(df[m].max())})
    res_df = pd.DataFrame(res).sort_values("pearson_r", ascending=False)
    res_df.to_csv(RES / "q4_metric_correlations.csv", index=False)
    print("\n=== Metric vs decoder_corr ===")
    print(res_df.round(4).to_string(index=False))

    best_metric = res_df.iloc[(res_df["pearson_r"].abs().argmax())]
    print(f"\nBest metric (|r| largest): {best_metric['metric']}  "
          f"r = {best_metric['pearson_r']:+.3f}  p = {best_metric['p_value']:.3g}")

    # ----- Plot -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: canonical correlation decay curve, mean ± std across 156 pairs
    cc_cols = [f"cc_{i+1}" for i in range(K_PCS)]
    cc_matrix = df[cc_cols].to_numpy()  # (156, 15)
    mean_curve = cc_matrix.mean(axis=0)
    std_curve = cc_matrix.std(axis=0)
    x = np.arange(1, K_PCS + 1)
    ax = axes[0, 0]
    ax.plot(x, mean_curve, "o-", color="tab:blue", label="mean across 156 pairs")
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                    alpha=0.25, color="tab:blue", label="± 1 std")
    ax.axhline(EFFECTIVE_DIM_THRESHOLD, color="grey", linestyle=":",
               label=f"threshold = {EFFECTIVE_DIM_THRESHOLD}")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Canonical correlation")
    ax.set_title(f"A. Canonical correlation profile across 156 r1 pairs (top-{K_PCS})")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Panel B: bar chart of |r| for each metric
    ax = axes[0, 1]
    res_sorted = res_df.copy()
    res_sorted["abs_r"] = res_sorted["pearson_r"].abs()
    res_sorted = res_sorted.sort_values("abs_r", ascending=True)
    colors = ["tab:red" if r < 0 else "tab:green" for r in res_sorted["pearson_r"]]
    ax.barh(res_sorted["metric"], res_sorted["abs_r"], color=colors)
    for i, (mname, r, p) in enumerate(
        zip(res_sorted["metric"], res_sorted["pearson_r"], res_sorted["p_value"])
    ):
        ax.text(res_sorted["abs_r"].iloc[i] + 0.005, i,
                f"r={r:+.2f}, p={p:.2g}", va="center", fontsize=8)
    ax.set_xlabel("|Pearson r| with decoder cross-day corr")
    ax.set_title("B. Which alignment metric predicts decoder transfer?")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel C: scatter for the BEST metric
    best_m = best_metric["metric"]
    ax = axes[1, 0]
    ax.scatter(df[best_m], df["decoder_corr"], alpha=0.5, color="tab:green", s=22)
    slope, intercept, r, p, _ = stats.linregress(df[best_m], df["decoder_corr"])
    xx = np.linspace(df[best_m].min(), df[best_m].max(), 50)
    ax.plot(xx, slope * xx + intercept, "k--",
            label=f"slope={slope:+.3f}, r={r:+.2f}, p={p:.3g}")
    ax.set_xlabel(f"Alignment metric: {best_m}")
    ax.set_ylabel("Cross-day decoder correlation")
    ax.set_title(f"C. Best metric: {best_m}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel D: per-pair canonical corr profiles, colored by decoder corr
    ax = axes[1, 1]
    decoder_vals = df["decoder_corr"].to_numpy()
    norm = plt.Normalize(vmin=decoder_vals.min(), vmax=decoder_vals.max())
    cmap = plt.cm.viridis
    order = np.argsort(decoder_vals)  # plot worst first so best are on top
    for idx in order:
        ax.plot(x, cc_matrix[idx], color=cmap(norm(decoder_vals[idx])),
                alpha=0.3, linewidth=0.7)
    ax.axhline(EFFECTIVE_DIM_THRESHOLD, color="grey", linestyle=":")
    ax.set_xlabel("PC index")
    ax.set_ylabel("Canonical correlation")
    ax.set_title("D. Per-pair profiles, colored by decoder corr (yellow = high)")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="decoder cross-day corr")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Q4 follow-up: localizing where manifold alignment matters for decoder transfer")
    fig.tight_layout()
    out = RES / "q4_followup.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
