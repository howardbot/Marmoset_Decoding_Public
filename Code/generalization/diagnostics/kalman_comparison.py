"""Head-to-head comparison: Origin Kording Kalman vs custom trial-aware Kalman.

Run both Kalman implementations on the same 13 r1 sessions, same locked
configuration, same 5-fold-by-trial CV splits. Compute M1 (concat) and M2
(per-trial) for each session and fold, then compare paired.

Structural difference between the two implementations:
  - Origin Kording (Neural_Decoding.KalmanFilterRegression):
        fits state transition A on all consecutive (x_t, x_{t+1}) pairs,
        including pairs that span a trial boundary in the stacked X matrix.
  - Custom Kalman_filter.KalmanFilterDecoder:
        fits A only on within-trial (x_t, x_{t+1}) pairs (per-trial groupby),
        excluding cross-trial seams.

Both use C = 1.0 noise scaling and reset the Kalman state at the start of
each test trial during prediction.

Outputs:
  Results/workflows/generalization/kalman_comparison_per_session.csv
  Results/workflows/generalization/kalman_comparison_per_fold.csv
  Results/workflows/generalization/kalman_comparison.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from Neural_Decoding.decoders import KalmanFilterRegression

_THIS_DIR = Path(__file__).resolve().parents[1]
_CODE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_THIS_DIR))

import decoder_utils as du  # noqa: E402
from Kalman_filter import KalmanFilterDecoder  # noqa: E402  custom trial-aware
from cross_day_decoder import (  # noqa: E402
    build_session_cache_entry, list_sessions, session_date, session_epoch,
    compute_metric_set, load_optimal_lag_per_session,
    BIN_SIZE_MS,
)

warnings.filterwarnings("ignore")

RES = _THIS_DIR.parents[1] / "Results" / "workflows" / "generalization"
FIG_DIR = _THIS_DIR.parents[1] / "Results" / "archive" / "legacy" / "report_figures"
RES.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5


def preprocess_train_test(X_train, Y_train, X_test, Y_test):
    """Same Z-score / center logic as cross_day_decoder.kalman_fit_predict."""
    keep = np.nanstd(Y_train, axis=0) > 1e-12
    Y_train, Y_test = Y_train[:, keep], Y_test[:, keep]
    Y_mean = np.nanmean(Y_train, axis=0)
    Y_std = np.nanstd(Y_train, axis=0)
    X_mean = np.nanmean(X_train, axis=0)
    Y_train = (Y_train - Y_mean) / Y_std
    Y_test = (Y_test - Y_mean) / Y_std
    X_train = X_train - X_mean
    X_test_c = X_test - X_mean
    return X_train, Y_train, X_test_c, Y_test


def predict_origin(model, Y_test, X_test_c, meta_test):
    """Trial-by-trial predict using Origin Kording's KalmanFilterRegression."""
    pred = np.full_like(X_test_c, np.nan, dtype=float)
    for _, idx in meta_test.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        pred[idx] = np.asarray(model.predict(Y_test[idx], X_test_c[idx]), dtype=float)
    return pred


def predict_custom(model, Y_test, X_test_c, meta_test):
    """Trial-aware predict (already handles per-trial state reset internally)."""
    return np.asarray(model.predict(Y_test, X_test_c, meta_test), dtype=float)


def main():
    sessions = [t for t in list_sessions() if session_epoch(t) == "r1"]
    print(f"Building cache for {len(sessions)} r1 sessions ...")
    cache = {session_date(s): build_session_cache_entry(s) for s in sessions}

    lags = load_optimal_lag_per_session()
    default_lag = int(np.median(list(lags.values()))) if lags else 100

    per_fold_rows = []
    for date, entry in cache.items():
        X_full = entry["X"]
        Y_full = entry["Y_pc"]   # use PC-space Y so neuron counts match across sessions
        meta_full = entry["meta"]
        lag_ms = int(lags.get(date, default_lag))
        lag_bins = lag_ms // BIN_SIZE_MS

        X_lag, Y_lag, meta_lag = du.apply_lag(X_full, Y_full, meta_full, lag_bins, verbose=False)

        for fold_idx, (train_mask, test_mask) in enumerate(
            du.kfold_split_by_trial(meta_lag, n_splits=N_SPLITS)
        ):
            X_tr, Y_tr = X_lag[train_mask], Y_lag[train_mask]
            X_te, Y_te = X_lag[test_mask], Y_lag[test_mask]
            meta_tr = meta_lag[train_mask].reset_index(drop=True)
            meta_te = meta_lag[test_mask].reset_index(drop=True)

            X_tr_c, Y_tr_z, X_te_c, Y_te_z = preprocess_train_test(X_tr, Y_tr, X_te, Y_te)

            # Origin Kording
            model_o = KalmanFilterRegression(C=1)
            model_o.fit(Y_tr_z, X_tr_c)
            pred_o = predict_origin(model_o, Y_te_z, X_te_c, meta_te)
            m_o = compute_metric_set(X_te_c, pred_o, meta_te)

            # Custom trial-aware
            model_c = KalmanFilterDecoder(C=1.0)
            model_c.fit(Y_tr_z, X_tr_c, meta_tr)
            pred_c = predict_custom(model_c, Y_te_z, X_te_c, meta_te)
            m_c = compute_metric_set(X_te_c, pred_c, meta_te)

            per_fold_rows.append({
                "date": date,
                "fold": fold_idx,
                "lag_ms": lag_ms,
                "origin_M1": m_o["corr_concat_mean"],
                "origin_M2": m_o["corr_per_trial_mean"],
                "custom_M1": m_c["corr_concat_mean"],
                "custom_M2": m_c["corr_per_trial_mean"],
            })
        print(f"  {date}: done")

    pf = pd.DataFrame(per_fold_rows)
    pf.to_csv(RES / "kalman_comparison_per_fold.csv", index=False)

    per_sess = pf.groupby("date").agg(
        origin_M1=("origin_M1", "mean"),
        origin_M2=("origin_M2", "mean"),
        custom_M1=("custom_M1", "mean"),
        custom_M2=("custom_M2", "mean"),
    ).reset_index()
    per_sess.to_csv(RES / "kalman_comparison_per_session.csv", index=False)

    print("\n=== Per-session within-day mean (avg across folds) ===")
    print(per_sess.round(3).to_string(index=False))

    # ---- Stats ----
    diff_M1 = per_sess["custom_M1"] - per_sess["origin_M1"]
    diff_M2 = per_sess["custom_M2"] - per_sess["origin_M2"]
    print()
    print("=== Paired comparison across 13 sessions ===")
    print(f"  M1 (concat):")
    print(f"    Origin mean = {per_sess.origin_M1.mean():+.4f}  "
          f"Custom mean = {per_sess.custom_M1.mean():+.4f}  "
          f"Δ = {diff_M1.mean():+.4f}")
    w1, p1 = stats.wilcoxon(per_sess.origin_M1, per_sess.custom_M1)
    print(f"    Wilcoxon p = {p1:.3g} (2-sided)")
    print(f"  M2 (per-trial):")
    print(f"    Origin mean = {per_sess.origin_M2.mean():+.4f}  "
          f"Custom mean = {per_sess.custom_M2.mean():+.4f}  "
          f"Δ = {diff_M2.mean():+.4f}")
    w2, p2 = stats.wilcoxon(per_sess.origin_M2, per_sess.custom_M2)
    print(f"    Wilcoxon p = {p2:.3g} (2-sided)")
    r_M1, _ = stats.pearsonr(per_sess.origin_M1, per_sess.custom_M1)
    r_M2, _ = stats.pearsonr(per_sess.origin_M2, per_sess.custom_M2)
    print(f"  Pair-level Pearson r: M1 = {r_M1:+.3f}, M2 = {r_M2:+.3f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(per_sess))
    w = 0.35

    # A: per-session bar chart, both metrics
    ax = axes[0]
    ax.bar(x - w/2, per_sess.origin_M2, w, label="Origin (M2)", color="tab:orange")
    ax.bar(x + w/2, per_sess.custom_M2, w, label="Custom trial-aware (M2)", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(per_sess.date, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Within-day M2 per-trial corr (mean across 5 folds)")
    ax.set_title("A. Per session within-day M2 (n=13)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # B: paired scatter M2
    ax = axes[1]
    ax.scatter(per_sess.origin_M2, per_sess.custom_M2, s=60, alpha=0.7, color="tab:purple")
    for _, r in per_sess.iterrows():
        ax.annotate(str(r["date"])[-4:], (r.origin_M2, r.custom_M2),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    lo = min(per_sess.origin_M2.min(), per_sess.custom_M2.min()) - 0.02
    hi = max(per_sess.origin_M2.max(), per_sess.custom_M2.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Origin Kording M2")
    ax.set_ylabel("Custom trial-aware M2")
    ax.set_title("B. Paired (M2)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # C: paired scatter M1
    ax = axes[2]
    ax.scatter(per_sess.origin_M1, per_sess.custom_M1, s=60, alpha=0.7, color="tab:green")
    for _, r in per_sess.iterrows():
        ax.annotate(str(r["date"])[-4:], (r.origin_M1, r.custom_M1),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    lo = min(per_sess.origin_M1.min(), per_sess.custom_M1.min()) - 0.02
    hi = max(per_sess.origin_M1.max(), per_sess.custom_M1.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Origin Kording M1")
    ax.set_ylabel("Custom trial-aware M1")
    ax.set_title("C. Paired (M1)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Origin Kording Kalman vs custom trial-aware Kalman — within-day (locked config, 13 r1 sessions)")
    fig.tight_layout()
    out_results = RES / "kalman_comparison.png"
    out_figs = FIG_DIR / "kalman_comparison.png"
    fig.savefig(out_results, dpi=150)
    fig.savefig(out_figs, dpi=150)
    print(f"\nSaved {out_results}")
    print(f"Saved {out_figs}")


if __name__ == "__main__":
    main()
