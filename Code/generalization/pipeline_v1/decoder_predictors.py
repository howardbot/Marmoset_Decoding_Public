"""What predicts cross-day decoder transfer? (since manifold alignment doesn't)

For each of the 156 ordered r1 pairs, we compute a battery of features that
plausibly affect how well a decoder trained on day i transfers to day j, and
regress the actual decoder transfer correlation against them.

Feature families:
  - Training-side quality:   trial count, total spike count, within-day
                             5-fold CV decoder corr, between-trial kinematic
                             variance, mean unit firing rate
  - Test-side quality:       same set, evaluated on the test day
  - Pair-level similarity:   day gap (calendar days), trajectory similarity
                             (mean Pearson r between train and test average
                             reach trajectories, per kinematic dim)
  - Manifold alignment:      mean top-5 canonical correlation (already shown
                             to be saturated -> kept here as a control)

We report:
  - Each feature's univariate Pearson r with decoder transfer corr
  - A multivariate OLS fit using all features -> overall R^2 and per-feature betas
  - Plots showing top predictor and actual vs predicted

Outputs:
  Results/workflows/generalization/decoder_predictors_features.csv
  Results/workflows/generalization/decoder_predictors_univariate.csv
  Results/workflows/generalization/decoder_predictors.png
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
    build_session_cache_entry, list_sessions, session_date, session_epoch,
)
from manifold_align import heldout_canonical_correlations  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
RES = REPO_ROOT / "Results" / "workflows" / "generalization"
N_PHASE_BINS = 30


def trial_average_X(X, meta, n_phase_bins=N_PHASE_BINS):
    """Phase-resample each trial's velocity and average -> (n_phase_bins, 3)."""
    X = np.asarray(X, dtype=float)
    trial_arrays = []
    for _, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 3:
            continue
        t_data = np.linspace(0, 1, len(idx))
        t_targ = np.linspace(0, 1, n_phase_bins)
        trial_arrays.append(np.column_stack([
            np.interp(t_targ, t_data, X[idx, d]) for d in range(X.shape[1])
        ]))
    if not trial_arrays:
        return np.zeros((n_phase_bins, X.shape[1]))
    return np.mean(trial_arrays, axis=0)


def main():
    # ---- existing per-pair decoder corr from CSV ----
    long_df = pd.read_csv(RES / "cross_day_corr_long.csv")
    r1_df = long_df[(long_df.train_epoch == "r1") & (long_df.test_epoch == "r1")]
    metric_col = "corr_per_trial_mean" if "corr_per_trial_mean" in r1_df.columns else "corr"
    per_pair = (
        r1_df.groupby(["train_date", "test_date"])[metric_col]
        .mean().reset_index(name="decoder_corr")
    )
    dates = sorted(per_pair.train_date.unique())

    # ---- per-session features (need fresh cache for X, meta, traj) ----
    print(f"Building cache for {len(dates)} r1 sessions ...")
    cache = {}
    for tag in [t for t in list_sessions() if session_epoch(t) == "r1"]:
        d = session_date(tag)
        cache[d] = build_session_cache_entry(tag)
    print("Cache ready.")

    per_session = {}
    for d, entry in cache.items():
        X = entry["X"]
        meta = entry["meta"]
        per_session[d] = {
            "n_trials": int(meta["trial_number"].nunique()),
            "n_bins": int(X.shape[0]),
            "n_units": int(entry["PCA_V"].shape[0]),
            "kin_total_var": float(np.sum(np.var(X, axis=0))),
            "kin_between_trial_var": float(np.sum(np.var(np.stack([
                X[np.asarray(idx)].mean(axis=0) for _, idx in meta.groupby("trial_number").indices.items()
                if len(idx) > 0
            ]), axis=0))),
            "avg_traj_X": trial_average_X(X, meta),
        }

    # within-day corrs from generalization_summary.csv
    summary = pd.read_csv(RES / "generalization_summary.csv")
    summary_r1 = summary[summary.train_epoch == "r1"].set_index("train_date")
    for d in dates:
        per_session[str(int(d))]["within_day_corr"] = float(summary_r1.loc[d, "within_day"])

    # ---- per-pair features ----
    feat_rows = []
    for _, row in per_pair.iterrows():
        d_tr, d_te = str(int(row.train_date)), str(int(row.test_date))
        if d_tr == d_te:
            continue
        a = per_session[d_tr]
        b = per_session[d_te]
        # trajectory similarity: mean Pearson r across 3 vel dims
        traj_corrs = []
        for dim in range(3):
            if np.std(a["avg_traj_X"][:, dim]) == 0 or np.std(b["avg_traj_X"][:, dim]) == 0:
                continue
            traj_corrs.append(float(np.corrcoef(a["avg_traj_X"][:, dim], b["avg_traj_X"][:, dim])[0, 1]))
        # manifold alignment (top-5) as a kept control — held-out CC (in-sample saturates)
        cc = heldout_canonical_correlations(cache[d_tr], cache[d_te])
        mean_top5_cc = float(np.mean(cc[:5]))
        gap = abs((pd.to_datetime(d_tr, format="%Y%m%d") - pd.to_datetime(d_te, format="%Y%m%d")).days)
        feat_rows.append({
            "train_date": row.train_date,
            "test_date": row.test_date,
            "decoder_corr": float(row.decoder_corr),
            # train-side
            "train_n_trials": a["n_trials"],
            "train_n_bins": a["n_bins"],
            "train_n_units": a["n_units"],
            "train_within_day_corr": a["within_day_corr"],
            "train_kin_total_var": a["kin_total_var"],
            "train_kin_between_trial_var": a["kin_between_trial_var"],
            # test-side
            "test_n_trials": b["n_trials"],
            "test_n_bins": b["n_bins"],
            "test_n_units": b["n_units"],
            "test_within_day_corr": b["within_day_corr"],
            "test_kin_total_var": b["kin_total_var"],
            "test_kin_between_trial_var": b["kin_between_trial_var"],
            # pair-level
            "day_gap": gap,
            "traj_similarity": float(np.mean(traj_corrs)) if traj_corrs else np.nan,
            "manifold_top5_cc": mean_top5_cc,
        })
    feats = pd.DataFrame(feat_rows)
    feats.to_csv(RES / "decoder_predictors_features.csv", index=False)
    print(f"Saved per-pair feature matrix ({len(feats)} rows)")

    feature_cols = [c for c in feats.columns if c not in
                    ("train_date", "test_date", "decoder_corr")]

    # ---- Univariate correlations ----
    rows = []
    for c in feature_cols:
        v = feats[c].to_numpy()
        y = feats["decoder_corr"].to_numpy()
        good = np.isfinite(v) & np.isfinite(y)
        if good.sum() < 5 or np.std(v[good]) == 0:
            rows.append({"feature": c, "r": np.nan, "p": np.nan})
            continue
        r, p = stats.pearsonr(v[good], y[good])
        rows.append({"feature": c, "r": r, "p": p, "abs_r": abs(r)})
    uni = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    uni.to_csv(RES / "decoder_predictors_univariate.csv", index=False)
    print("\n=== Univariate Pearson r with decoder_corr ===")
    print(uni[["feature", "r", "p"]].round(4).to_string(index=False))

    # ---- Multivariate OLS ----
    X_mat = feats[feature_cols].to_numpy()
    y = feats["decoder_corr"].to_numpy()
    # z-score features so betas are comparable
    means = X_mat.mean(axis=0)
    stds = X_mat.std(axis=0)
    stds[stds == 0] = 1.0
    Xz = (X_mat - means) / stds
    Xz_aug = np.column_stack([np.ones(len(Xz)), Xz])
    coefs, *_ = np.linalg.lstsq(Xz_aug, y, rcond=None)
    y_pred = Xz_aug @ coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\nMultivariate OLS (all features, z-scored): R^2 = {r2:.3f}")
    print("Standardized betas (sorted by |beta|):")
    betas = pd.DataFrame({
        "feature": feature_cols,
        "beta_std": coefs[1:],
    }).assign(abs_beta=lambda d: np.abs(d.beta_std)).sort_values("abs_beta", ascending=False)
    print(betas.round(4).to_string(index=False))

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A: univariate |r| bar chart
    ax = axes[0, 0]
    uni_sorted = uni.dropna(subset=["abs_r"]).sort_values("abs_r", ascending=True)
    colors = ["tab:red" if r < 0 else "tab:green" for r in uni_sorted["r"]]
    ax.barh(uni_sorted["feature"], uni_sorted["abs_r"], color=colors)
    for i, (f_, r_, p_) in enumerate(zip(uni_sorted["feature"], uni_sorted["r"], uni_sorted["p"])):
        sig = "*" if (np.isfinite(p_) and p_ < 0.05) else ""
        ax.text(uni_sorted["abs_r"].iloc[i] + 0.005, i,
                f"r={r_:+.2f}, p={p_:.2g}{sig}", va="center", fontsize=8)
    ax.set_xlabel("|Pearson r| with decoder cross-day corr")
    ax.set_title("A. Univariate predictors of cross-day decoder corr (156 r1 pairs)")
    ax.grid(True, alpha=0.3, axis="x")

    # B: scatter for top single predictor
    top_feat = uni_sorted.iloc[-1]["feature"]
    ax = axes[0, 1]
    v = feats[top_feat].to_numpy()
    y_arr = feats["decoder_corr"].to_numpy()
    ax.scatter(v, y_arr, alpha=0.55, s=22, color="tab:green")
    slope, intercept, r_, p_, _ = stats.linregress(v, y_arr)
    xx = np.linspace(v.min(), v.max(), 50)
    ax.plot(xx, slope * xx + intercept, "k--",
            label=f"r={r_:+.2f}, p={p_:.3g}")
    ax.set_xlabel(top_feat)
    ax.set_ylabel("Cross-day decoder corr")
    ax.set_title(f"B. Top single predictor: {top_feat}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # C: multivariate predicted vs actual
    ax = axes[1, 0]
    ax.scatter(y_pred, y, alpha=0.55, s=22, color="tab:purple")
    lo, hi = min(y_pred.min(), y.min()), max(y_pred.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Predicted (multivariate OLS)")
    ax.set_ylabel("Actual decoder corr")
    ax.set_title(f"C. Multivariate OLS fit: R^2 = {r2:.3f}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # D: standardized betas
    ax = axes[1, 1]
    bet_sorted = betas.sort_values("abs_beta", ascending=True)
    colors = ["tab:red" if b < 0 else "tab:green" for b in bet_sorted["beta_std"]]
    ax.barh(bet_sorted["feature"], bet_sorted["abs_beta"], color=colors)
    for i, (f_, b_) in enumerate(zip(bet_sorted["feature"], bet_sorted["beta_std"])):
        ax.text(bet_sorted["abs_beta"].iloc[i] + 0.005, i,
                f"β={b_:+.2f}", va="center", fontsize=8)
    ax.set_xlabel("|standardized β| in joint OLS")
    ax.set_title("D. Multivariate β (green +, red −)")
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("What predicts cross-day decoder transfer? (r1 internal, n=156 pairs)")
    fig.tight_layout()
    out = RES / "decoder_predictors.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
