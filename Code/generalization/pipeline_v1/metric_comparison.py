"""Validate that M1 (concat corr) and M2 (per-trial corr) give consistent rankings.

We re-run the manifold-aligned cross-day pipeline for all 169 cells and report
both metrics side by side. If their pair-level Pearson r is high (> 0.7),
choice of metric does not move the qualitative story.

Outputs:
  Results/workflows/generalization/metric_comparison_per_pair.csv
  Results/workflows/generalization/metric_comparison.png
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
    kalman_fit_predict, compute_metric_set, load_optimal_lag_per_session,
    BIN_SIZE_MS, N_SPLITS_DIAGONAL,
)
from manifold_align import cca_align, apply_alignment  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
RES = REPO_ROOT / "Results" / "workflows" / "generalization"


def eval_off_diagonal_metrics(train_data, test_data, lag_bins):
    W_tr, W_te, m_tr, m_te = cca_align(train_data["traj"], test_data["traj"])
    Y_tr_canon = apply_alignment(train_data["Y_pc"], W_tr, m_tr)
    Y_te_canon = apply_alignment(test_data["Y_pc"], W_te, m_te)
    X_tr_lag, Y_tr_lag, _ = du.apply_lag(
        train_data["X"], Y_tr_canon, train_data["meta"], lag_bins, verbose=False
    )
    X_te_lag, Y_te_lag, meta_te_lag = du.apply_lag(
        test_data["X"], Y_te_canon, test_data["meta"], lag_bins, verbose=False
    )
    X_te_c, pred = kalman_fit_predict(
        X_tr_lag, Y_tr_lag, X_te_lag, Y_te_lag, meta_te_lag
    )
    return compute_metric_set(X_te_c, pred, meta_te_lag)


def eval_diagonal_metrics(data, lag_bins):
    X_lag, Y_lag, meta_lag = du.apply_lag(
        data["X"], data["Y_pc"], data["meta"], lag_bins, verbose=False
    )
    per_fold = []
    for train_mask, test_mask in du.kfold_split_by_trial(meta_lag, n_splits=N_SPLITS_DIAGONAL):
        X_tr, Y_tr = X_lag[train_mask], Y_lag[train_mask]
        X_te, Y_te = X_lag[test_mask], Y_lag[test_mask]
        meta_te = meta_lag[test_mask].reset_index(drop=True)
        X_te_c, pred = kalman_fit_predict(X_tr, Y_tr, X_te, Y_te, meta_te)
        per_fold.append(compute_metric_set(X_te_c, pred, meta_te))
    return {
        "corr_concat_mean": float(np.nanmean([f["corr_concat_mean"] for f in per_fold])),
        "corr_per_trial_mean": float(np.nanmean([f["corr_per_trial_mean"] for f in per_fold])),
    }


def main():
    sessions = [t for t in list_sessions() if session_epoch(t) == "r1"]
    print(f"Building cache for {len(sessions)} r1 sessions ...")
    cache = {session_date(s): build_session_cache_entry(s) for s in sessions}
    dates = sorted(cache.keys())

    lags = load_optimal_lag_per_session()
    default_lag = int(np.median(list(lags.values()))) if lags else 100

    rows = []
    n = len(dates)
    for i, d_tr in enumerate(dates):
        lag_ms = int(lags.get(d_tr, default_lag))
        lag_bins = lag_ms // BIN_SIZE_MS
        for j, d_te in enumerate(dates):
            try:
                if i == j:
                    m = eval_diagonal_metrics(cache[d_tr], lag_bins)
                    kind = "diagonal"
                else:
                    m = eval_off_diagonal_metrics(cache[d_tr], cache[d_te], lag_bins)
                    kind = "off_diagonal"
            except Exception as e:
                print(f"  FAIL {d_tr}->{d_te}: {type(e).__name__}: {e}")
                m = {"corr_concat_mean": np.nan, "corr_per_trial_mean": np.nan}
                kind = "fail"
            rows.append({"train_date": d_tr, "test_date": d_te, "kind": kind, **m})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "metric_comparison_per_pair.csv", index=False)
    print(f"Saved {RES / 'metric_comparison_per_pair.csv'}  ({len(df)} pairs)")

    off = df[df.kind == "off_diagonal"]
    print("\n=== Off-diagonal summary (n=156) ===")
    for m in ("corr_concat_mean", "corr_per_trial_mean"):
        print(f"  {m}: mean={off[m].mean():+.3f}  median={off[m].median():+.3f}  "
              f"range [{off[m].min():+.3f}, {off[m].max():+.3f}]")
    r, p = stats.pearsonr(off["corr_concat_mean"], off["corr_per_trial_mean"])
    print(f"\nM1 vs M2 pair-level Pearson r = {r:+.3f}, p = {p:.3g}")

    # ===== Plots =====
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Scatter M1 vs M2
    ax = axes[0]
    ax.scatter(off["corr_concat_mean"], off["corr_per_trial_mean"],
               alpha=0.55, s=22, color="tab:purple")
    lo = min(off["corr_concat_mean"].min(), off["corr_per_trial_mean"].min())
    hi = max(off["corr_concat_mean"].max(), off["corr_per_trial_mean"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("M1 concat corr")
    ax.set_ylabel("M2 per-trial corr")
    ax.set_title(f"M1 vs M2 (off-diagonal, n=156)\nPearson r = {r:+.2f}, p = {p:.2g}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Distribution overlay
    ax = axes[1]
    bins = np.linspace(-0.05, 0.45, 30)
    ax.hist(off["corr_concat_mean"].dropna(), bins=bins, alpha=0.6, color="tab:orange",
            label=f"M1 concat  μ={off['corr_concat_mean'].mean():+.2f}")
    ax.hist(off["corr_per_trial_mean"].dropna(), bins=bins, alpha=0.6, color="tab:blue",
            label=f"M2 per-trial  μ={off['corr_per_trial_mean'].mean():+.2f}")
    ax.axvline(0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Mean velocity correlation")
    ax.set_ylabel("Number of off-diagonal pairs")
    ax.set_title("Distribution overlay")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("M1 (concat) vs M2 (per-trial) on the same r1 cross-day Kalman predictions")
    fig.tight_layout()
    out = RES / "metric_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
