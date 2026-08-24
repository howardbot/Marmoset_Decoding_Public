"""Smoke test: within-day on 0813 with the new Butterworth trajectory smoother.

Times the core operations so we can extrapolate the full sweep cost on this
M2 machine. Not part of the main pipeline -- safe to delete once the full
sweep script is ready.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

import decoder_utils as du
from Neural_Decoding.decoders import KalmanFilterRegression, WienerFilterRegression
from Wiener_filter import history_bins_for

warnings.filterwarnings("ignore")

SESSION = "TSAL20250813_0830_staticAndStaticFree001"
BIN_MS = 30
BIN_S = BIN_MS / 1000.0
TARGET = "relative_velocity"
UNIT_QUALITIES = ("good", "mua")
SMOOTH_SIGMA_MS = 50
LAGS_MS = [0, 30, 60, 90, 120, 150]   # nearest bin-integer in [0,150] for 30ms bin
N_FOLDS = 5
HISTORY_MS_LIST = [50, 100]


def corr_1d(y_true, y_pred):
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if good.sum() < 2:
        return np.nan
    yt = np.asarray(y_true[good], dtype=float)
    yp = np.asarray(y_pred[good], dtype=float)
    if np.std(yt) == 0 or np.std(yp) == 0:
        return np.nan
    return float(np.corrcoef(yt, yp)[0, 1])


def m2_per_trial(X_te, pred, meta_te):
    """M2 metric: per-trial Pearson r, averaged across trials and 3 dims."""
    vals = []
    for _, idx in meta_te.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 4:
            continue
        for d in range(X_te.shape[1]):
            vals.append(corr_1d(X_te[idx, d], pred[idx, d]))
    return float(np.nanmean(vals)) if vals else np.nan


def run_one(smoother_label, smoother_kwargs):
    """Build dataset with given trajectory smoother, run 5-fold CV across lags
    for both Kalman and Wiener; return list of result rows + timing breakdown."""
    rows = []
    timings = {}

    du.SESSION = SESSION
    du.PROCESSED_NWB = du.DATA_DIR / f"{SESSION}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_S

    t0 = time.time()
    io, nwb_prc, reach_tbl = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb_prc, reach_tbl, TARGET,
            bin_size=BIN_S,
            unit_qualities=UNIT_QUALITIES,
            trial_window="start_to_peak",
            **smoother_kwargs,
        )
    finally:
        io.close()
    timings["build_dataset_s"] = time.time() - t0

    # Neural causal Gaussian smoothing (sigma=50ms).
    t0 = time.time()
    sigma_bins = SMOOTH_SIGMA_MS / BIN_MS
    Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=sigma_bins)
    timings["neural_smooth_s"] = time.time() - t0

    timings["per_fold_kalman_s"] = 0.0
    timings["per_fold_wiener_s"] = 0.0
    n_kalman_fits = 0
    n_wiener_fits = 0

    # Snapshot the 5 fold masks once (yielded twice through the generator otherwise).
    fold_masks = list(du.kfold_split_by_trial(meta, n_splits=N_FOLDS, random_seed=0))

    for lag_ms in LAGS_MS:
        lag_bins = lag_ms // BIN_MS

        for fold_idx, (tr_mask, te_mask) in enumerate(fold_masks):
            # Split on the original (un-lagged) meta, then apply per-trial lag on
            # each side. apply_lag operates trial-wise so the split is preserved.
            X_tr_lag, Y_tr_lag, meta_tr_lag = du.apply_lag(
                X[tr_mask], Y_sm[tr_mask], meta[tr_mask].reset_index(drop=True),
                lag_bins, verbose=False,
            )
            X_te_lag, Y_te_lag, meta_te_lag = du.apply_lag(
                X[te_mask], Y_sm[te_mask], meta[te_mask].reset_index(drop=True),
                lag_bins, verbose=False,
            )
            X_tr, Y_tr = X_tr_lag, Y_tr_lag
            X_te, Y_te = X_te_lag, Y_te_lag
            meta_te = meta_te_lag
            if len(X_tr) < 50 or len(X_te) < 20:
                continue

            # --- Kalman ---
            t0 = time.time()
            keep = np.nanstd(Y_tr, axis=0) > 1e-12
            Yk_tr, Yk_te = Y_tr[:, keep], Y_te[:, keep]
            Y_mean = np.nanmean(Yk_tr, axis=0)
            Y_std = np.nanstd(Yk_tr, axis=0)
            X_mean = np.nanmean(X_tr, axis=0)
            Yk_tr = (Yk_tr - Y_mean) / Y_std
            Yk_te = (Yk_te - Y_mean) / Y_std
            X_tr_c = X_tr - X_mean
            X_te_c = X_te - X_mean

            kal = KalmanFilterRegression(C=1)
            kal.fit(Yk_tr, X_tr_c)
            pred_kal = np.full_like(X_te_c, np.nan, dtype=float)
            for _, idx in meta_te.groupby("trial_number").indices.items():
                idx = np.asarray(idx)
                pred_kal[idx] = np.asarray(kal.predict(Yk_te[idx], X_te_c[idx]), dtype=float)
            m2_k = m2_per_trial(X_te_c, pred_kal, meta_te)
            timings["per_fold_kalman_s"] += time.time() - t0
            n_kalman_fits += 1
            rows.append({
                "smoother": smoother_label, "decoder": "kalman",
                "lag_ms": lag_ms, "fold": fold_idx,
                "history_ms": None, "M2": m2_k,
            })

            # --- Wiener (sweeps history_ms) ---
            for history_ms in HISTORY_MS_LIST:
                t0 = time.time()
                hb = history_bins_for(history_ms, BIN_MS)
                F_tr, Xh_tr, meta_tr_h = du.make_history_features(
                    X_tr, Y_tr, meta_tr_lag, hb, lag_bins=0,
                )
                F_te, Xh_te, meta_te_h = du.make_history_features(
                    X_te, Y_te, meta_te_lag, hb, lag_bins=0,
                )
                if len(F_tr) == 0 or len(F_te) == 0:
                    continue
                F_mean = np.nanmean(F_tr, axis=0)
                X_mean_w = np.nanmean(Xh_tr, axis=0)
                wmod = WienerFilterRegression()
                wmod.fit(F_tr - F_mean, Xh_tr - X_mean_w)
                pred_w = wmod.predict(F_te - F_mean) + X_mean_w
                m2_w = m2_per_trial(Xh_te, pred_w, meta_te_h)
                timings["per_fold_wiener_s"] += time.time() - t0
                n_wiener_fits += 1
                rows.append({
                    "smoother": smoother_label, "decoder": "wiener",
                    "lag_ms": lag_ms, "fold": fold_idx,
                    "history_ms": history_ms, "M2": m2_w,
                })

    timings["n_kalman_fits"] = n_kalman_fits
    timings["n_wiener_fits"] = n_wiener_fits
    if n_kalman_fits:
        timings["avg_kalman_fit_s"] = timings["per_fold_kalman_s"] / n_kalman_fits
    if n_wiener_fits:
        timings["avg_wiener_fit_s"] = timings["per_fold_wiener_s"] / n_wiener_fits
    return rows, timings


def summarise(rows, label):
    """Mean M2 per (decoder, lag, history)."""
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby(["decoder", "lag_ms", "history_ms"], dropna=False)["M2"].agg(["mean", "std", "count"]).reset_index()
    g = g.sort_values(["decoder", "lag_ms", "history_ms"], na_position="first")
    print(f"\n--- {label}: M2 across 5 folds (mean ± std) ---")
    print(g.to_string(index=False))


def main():
    overall_t0 = time.time()
    smoothers = [
        ("savgol_default", {"smoother": "savgol"}),
        ("butter_order2",  {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}),
        ("butter_order4",  {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 4}),
    ]
    all_rows = []
    all_timings = {}
    for label, kw in smoothers:
        print(f"\n=== running {label} ===")
        t0 = time.time()
        rows, timings = run_one(label, kw)
        all_rows.extend(rows)
        all_timings[label] = timings
        print(f"  elapsed: {time.time() - t0:.1f}s")

    print("\n=== TIMING ===")
    for label, tt in all_timings.items():
        print(f"  {label}:")
        for k, v in tt.items():
            print(f"    {k}: {v}")

    summarise(all_rows, "0813 within-day")
    print(f"\nTOTAL: {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
