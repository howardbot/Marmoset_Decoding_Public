"""CCA ablation: does manifold alignment actually move the needle?

Method 1 (full pipeline, already computed):
    PCA per day -> CCA align test PC trajectory to train PC trajectory ->
    Kalman in shared canonical space.

Method 2 (this script):
    PCA per day, NO CCA. Train Kalman in train's own PC space; feed test data
    in test's own PC space straight into that Kalman. Geometrically nonsensical
    if the per-day PC axes don't happen to line up by index, so any non-zero
    correlation here is a lower-bound "what if we did nothing smart."

We compute the off-diagonal entries for both methods and compare. Diagonal is
not informative (CCA between identical days is approximately identity).

Outputs:
  Results/workflows/generalization/cca_ablation_matrix.csv
  Results/workflows/generalization/cca_ablation.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR))

import decoder_utils as du  # noqa: E402
from cross_day_decoder import (  # noqa: E402
    build_session_cache_entry, list_sessions, session_date, session_epoch,
    kalman_fit_predict, corr_1d, compute_metric_set,
    load_optimal_lag_per_session, BIN_SIZE_MS,
)

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
RES = REPO_ROOT / "Results" / "workflows" / "generalization"


def eval_no_cca(train_data, test_data, lag_bins):
    """Off-diagonal eval without CCA: each day stays in its own PC space.

    Returns dict with both M1 (concat) and M2 (per-trial) so it can be compared
    to whichever metric is primary downstream.
    """
    Y_tr = train_data["Y_pc"]
    Y_te = test_data["Y_pc"]
    X_tr_lag, Y_tr_lag, meta_tr_lag = du.apply_lag(
        train_data["X"], Y_tr, train_data["meta"], lag_bins, verbose=False,
    )
    X_te_lag, Y_te_lag, meta_te_lag = du.apply_lag(
        test_data["X"], Y_te, test_data["meta"], lag_bins, verbose=False,
    )
    X_te_c, pred = kalman_fit_predict(X_tr_lag, Y_tr_lag, X_te_lag, Y_te_lag, meta_te_lag)
    return compute_metric_set(X_te_c, pred, meta_te_lag)


def main():
    sessions = [t for t in list_sessions() if session_epoch(t) == "r1"]
    print(f"Building cache for {len(sessions)} r1 sessions ...")
    cache = {session_date(s): build_session_cache_entry(s) for s in sessions}
    dates = sorted(cache.keys())
    print("Cache ready.")

    lags = load_optimal_lag_per_session()
    default_lag_ms = int(np.median(list(lags.values()))) if lags else 100

    # Existing CCA-on (method 1) numbers from CSV — read both metric variants
    long_df = pd.read_csv(RES / "cross_day_corr_long.csv")
    r1_pairs = long_df[(long_df.train_epoch == "r1") & (long_df.test_epoch == "r1")]
    primary_col = (
        "corr_per_trial_mean"
        if "corr_per_trial_mean" in r1_pairs.columns
        else "corr"
    )
    m1 = (
        r1_pairs.groupby(["train_date", "test_date"])[primary_col]
        .mean().reset_index(name="m1_corr")
    )
    m1["train_date"] = m1["train_date"].astype(int).astype(str)
    m1["test_date"] = m1["test_date"].astype(int).astype(str)

    # Compute method-2 (no CCA) for the off-diagonal cells, save the same
    # primary metric so the comparison is apples-to-apples
    records = []
    n = len(dates)
    for i, d_tr in enumerate(dates):
        lag_ms = int(lags.get(d_tr, default_lag_ms))
        lag_bins = lag_ms // BIN_SIZE_MS
        for j, d_te in enumerate(dates):
            if i == j:
                continue
            try:
                metrics = eval_no_cca(cache[d_tr], cache[d_te], lag_bins)
                m2 = metrics["corr_per_trial_mean"] if primary_col == "corr_per_trial_mean" else metrics["corr_concat_mean"]
            except Exception as e:
                print(f"  FAILED {d_tr}->{d_te}: {type(e).__name__}: {e}")
                m2 = np.nan
            records.append({"train_date": d_tr, "test_date": d_te, "m2_corr": m2})
    m2_df = pd.DataFrame(records)

    merged = m1.merge(m2_df, on=["train_date", "test_date"], how="inner")
    merged["delta_m1_minus_m2"] = merged["m1_corr"] - merged["m2_corr"]
    merged.to_csv(RES / "cca_ablation_matrix.csv", index=False)

    print("\n=== Summary across 156 r1 off-diagonal pairs ===")
    print(f"  method 1 (CCA): mean = {merged['m1_corr'].mean():.3f}, "
          f"median = {merged['m1_corr'].median():.3f}")
    print(f"  method 2 (no CCA): mean = {merged['m2_corr'].mean():.3f}, "
          f"median = {merged['m2_corr'].median():.3f}")
    print(f"  mean Δ (m1 − m2) = {merged['delta_m1_minus_m2'].mean():+.3f}")
    from scipy import stats
    diff = (merged['m1_corr'] - merged['m2_corr']).dropna().to_numpy()
    w, p_w = stats.wilcoxon(diff)
    print(f"  paired Wilcoxon (2-sided) p = {p_w:.3g}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # A: histogram overlay
    ax = axes[0]
    bins = np.linspace(-0.1, 0.4, 30)
    ax.hist(merged["m2_corr"].dropna(), bins=bins, alpha=0.55, color="grey",
            label=f"method 2 (no CCA)\nmean={merged['m2_corr'].mean():+.3f}")
    ax.hist(merged["m1_corr"].dropna(), bins=bins, alpha=0.65, color="tab:orange",
            label=f"method 1 (CCA)\nmean={merged['m1_corr'].mean():+.3f}")
    ax.axvline(0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Cross-day decoder corr")
    ax.set_ylabel("Number of pairs")
    ax.set_title("A. CCA vs no-CCA distributions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # B: scatter m1 vs m2
    ax = axes[1]
    ax.scatter(merged["m2_corr"], merged["m1_corr"], alpha=0.55, s=22, color="tab:purple")
    lo = min(merged["m1_corr"].min(), merged["m2_corr"].min())
    hi = max(merged["m1_corr"].max(), merged["m2_corr"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Method 2 (no CCA)")
    ax.set_ylabel("Method 1 (with CCA)")
    ax.set_title(f"B. Paired comparison  (Wilcoxon p = {p_w:.3g})")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # C: paired delta distribution
    ax = axes[2]
    ax.hist(diff, bins=20, color="tab:green", alpha=0.7)
    ax.axvline(0, color="k", linewidth=1)
    ax.axvline(diff.mean(), color="tab:red", linewidth=1, linestyle="--",
               label=f"mean Δ = {diff.mean():+.3f}")
    ax.set_xlabel("Δ = method 1 − method 2  (per pair)")
    ax.set_ylabel("Number of pairs")
    ax.set_title("C. Per-pair improvement from CCA alignment")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("CCA ablation: does manifold alignment actually help cross-day decoding?")
    fig.tight_layout()
    out = RES / "cca_ablation.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
