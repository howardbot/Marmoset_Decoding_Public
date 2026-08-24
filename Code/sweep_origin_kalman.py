"""Sweep Origin Kording Kalman across all sessions, bin sizes, target modes, and lags.

Configuration grid:
  - session: every TSAL*_processed.nwb in Data/
  - bin_size_ms: {10, 20}
  - target_mode: {relative_velocity, relative_position_velocity_acceleration}
  - lag_ms: 0..160 ms in steps of bin_size_ms (neural always leads kinematics;
    negative lag has no physical meaning here)
  - neural smoothing: causal Gaussian, sigma in {50, 100} ms swept as a
    comparison dimension; kernel width in bins = sigma_ms / bin_size_ms so the
    temporal scale is constant across bin sizes

What we report:
  - Per fold / per dim correlation (saved in a long CSV)
  - Per (session, bin_size, target_mode, lag) mean correlation
  - Per (session, bin_size, target_mode): the best lag and its mean correlation
  - Trend figure: x = session date, y = best-lag mean velocity correlation,
    one line per (bin_size, target_mode) combination (4 lines total).

Notes:
  - We compare configurations on the mean of the 3 velocity dimensions because
    they exist in both target modes; pos+vel+acc mean across all 9 dims would
    not be apples-to-apples with velocity-only.
  - We monkey-patch decoder_utils.SESSION / PROCESSED_NWB per iteration because
    the existing helpers read these module-level constants.
  - Run-side note: this script is currently write-only. Execute with
      python Code/sweep_origin_kalman.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Neural_Decoding.decoders import KalmanFilterRegression

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # Code/ for local imports
import decoder_utils as du

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Data"
RESULTS_DIR = REPO_ROOT / "Results"
FIG_DIR = REPO_ROOT / "Results" / "legacy" / "diagnostics" / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_LONG = RESULTS_DIR / "kalman_sweep_long.csv"
CSV_SUMMARY = RESULTS_DIR / "kalman_sweep_summary.csv"
FIG_TREND = FIG_DIR / "kalman_sweep_trend.png"

UNIT_QUALITIES = ("good", "mua")
TRIAL_WINDOW = "start_to_peak"
N_SPLITS = 5

BIN_SIZES_MS = [10, 20]
TARGET_MODES = ["relative_velocity", "relative_position_velocity_acceleration"]
LAG_MAX_MS = 160                   # inclusive
SMOOTH_SIGMAS_MS = [50, 100]       # causal Gaussian sigmas applied to neural counts


def list_sessions():
    """Return sorted session tags (filename minus the _processed.nwb suffix)."""
    return sorted(
        p.name.replace("_processed.nwb", "")
        for p in DATA_DIR.glob("*_processed.nwb")
    )


def session_date(session_tag):
    """Pull the YYYYMMDD date string out of a TSAL session tag."""
    return session_tag.split("_")[0].replace("TSAL", "")


def corr_score_1d(y_true, y_pred):
    """Pearson r between two 1D arrays, NaN-safe."""
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if good.sum() < 2:
        return np.nan
    y_true = np.asarray(y_true[good], dtype=float)
    y_pred = np.asarray(y_pred[good], dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def fit_predict_kalman(X_train, Y_train, X_test, Y_test, meta_test):
    """One fold of Origin Kording Kalman: preprocess, fit, predict trial-by-trial."""
    # drop units with zero training variance (cannot z-score)
    keep = np.nanstd(Y_train, axis=0) > 1e-12
    Y_train, Y_test = Y_train[:, keep], Y_test[:, keep]

    Y_mean, Y_std = np.nanmean(Y_train, axis=0), np.nanstd(Y_train, axis=0)
    X_mean = np.nanmean(X_train, axis=0)

    Y_train = (Y_train - Y_mean) / Y_std
    Y_test = (Y_test - Y_mean) / Y_std
    X_train = X_train - X_mean
    X_test = X_test - X_mean

    model = KalmanFilterRegression(C=1)
    model.fit(Y_train, X_train)

    pred = np.full_like(X_test, np.nan, dtype=float)
    # Kalman state must be re-initialized per trial; iterate trial-grouped indices.
    for _, idx in meta_test.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        pred[idx] = np.asarray(model.predict(Y_test[idx], X_test[idx]), dtype=float)
    return X_test, pred


def build_for_session(nwb_prc, reach_tbl, bin_size_s, target_mode):
    """Build the (X, Y, meta) dataset at a given bin size and target mode."""
    return du.build_decoder_dataset(
        nwb_prc,
        reach_tbl,
        target_mode,
        bin_size=bin_size_s,
        unit_qualities=UNIT_QUALITIES,
        trial_window=TRIAL_WINDOW,
    )


def evaluate_one(session_tag, bin_size_ms, target_mode, lag_ms_grid, smooth_sigmas_ms):
    """Run the full lag x smoothing sweep for one (session, bin_size, target_mode).

    Y is built once per (session, bin_size, target_mode); smoothing variants are
    derived from that same Y, and the apply_lag pass is repeated for each
    (smoothing, lag) combination. Returns long-format records.
    """
    bin_size_s = bin_size_ms / 1000.0
    # rebind module constants so load_nwb_and_reach picks up this session
    du.SESSION = session_tag
    du.PROCESSED_NWB = du.DATA_DIR / f"{session_tag}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_size_s  # only affects apply_lag's print of lag_ms

    io, nwb_prc, reach_tbl = du.load_nwb_and_reach()
    records = []
    try:
        X, Y, meta = build_for_session(nwb_prc, reach_tbl, bin_size_s, target_mode)
        names = du.state_names(target_mode)

        # Pre-smooth Y once per sigma so we don't redo convolution inside the lag loop.
        smoothed = {
            sigma_ms: du.smooth_neural_causal(Y, meta, sigma_bins=sigma_ms / bin_size_ms)
            for sigma_ms in smooth_sigmas_ms
        }

        for sigma_ms, Y_sm in smoothed.items():
            for lag_ms in lag_ms_grid:
                lag_bins = lag_ms // bin_size_ms
                X_lag, Y_lag, meta_lag = du.apply_lag(X, Y_sm, meta, lag_bins, verbose=False)
                for fold_idx, (train_mask, test_mask) in enumerate(
                    du.kfold_split_by_trial(meta_lag, n_splits=N_SPLITS)
                ):
                    X_tr, Y_tr = X_lag[train_mask], Y_lag[train_mask]
                    X_te, Y_te = X_lag[test_mask],  Y_lag[test_mask]
                    meta_te = meta_lag[test_mask].reset_index(drop=True)
                    X_te_c, pred = fit_predict_kalman(X_tr, Y_tr, X_te, Y_te, meta_te)
                    for d, name in enumerate(names):
                        records.append({
                            "session": session_tag,
                            "date": session_date(session_tag),
                            "bin_size_ms": bin_size_ms,
                            "target_mode": target_mode,
                            "smooth_sigma_ms": sigma_ms,
                            "lag_ms": lag_ms,
                            "fold": fold_idx,
                            "dim": name,
                            "corr": corr_score_1d(X_te_c[:, d], pred[:, d]),
                        })
    finally:
        io.close()
    return records


def summarize(long_df):
    """Collapse fold/dim into per-config summary used by the trend plot.

    Adds two metrics:
      mean_corr_all  - mean over all dims of the target_mode
      mean_corr_vel  - mean over the 3 velocity dims only (apples-to-apples)
    """
    vel_dims = {"vel_x", "vel_y", "vel_z"}
    vel_only = long_df[long_df["dim"].isin(vel_dims)]
    grp_keys = ["session", "date", "bin_size_ms", "target_mode", "smooth_sigma_ms", "lag_ms"]
    s_all = long_df.groupby(grp_keys)["corr"].mean().rename("mean_corr_all")
    s_vel = vel_only.groupby(grp_keys)["corr"].mean().rename("mean_corr_vel")
    return pd.concat([s_all, s_vel], axis=1).reset_index()


def best_lag_per_config(summary_df):
    """For each (session, bin_size, target_mode, smooth_sigma), pick the lag with best vel corr."""
    idx = summary_df.groupby(
        ["session", "date", "bin_size_ms", "target_mode", "smooth_sigma_ms"]
    )["mean_corr_vel"].idxmax()
    return summary_df.loc[idx].reset_index(drop=True)


def plot_trend(best_df, out_path):
    """Two panels (one per smoothing sigma); within each, 4 lines for bin x target."""
    best_df = best_df.copy()
    # parse YYYYMMDD as datetime so the x-axis spaces sessions by real calendar days
    best_df["date_dt"] = pd.to_datetime(best_df["date"].astype(str), format="%Y%m%d")
    sigmas = sorted(best_df["smooth_sigma_ms"].unique())
    fig, axes = plt.subplots(1, len(sigmas), figsize=(7 * len(sigmas), 4.5), sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    style = {
        (10, "relative_velocity"): ("o-", "tab:blue"),
        (10, "relative_position_velocity_acceleration"): ("s-", "tab:cyan"),
        (20, "relative_velocity"): ("o--", "tab:red"),
        (20, "relative_position_velocity_acceleration"): ("s--", "tab:orange"),
    }
    for ax, sigma in zip(axes, sigmas):
        for (bs, tm), (mk, color) in style.items():
            sub = best_df[
                (best_df.bin_size_ms == bs)
                & (best_df.target_mode == tm)
                & (best_df.smooth_sigma_ms == sigma)
            ].sort_values("date_dt")
            if sub.empty:
                continue
            label = f"{bs}ms · {'vel' if tm == 'relative_velocity' else 'pos+vel+acc'}"
            ax.plot(
                sub["date_dt"], sub["mean_corr_vel"], mk, color=color,
                label=label, linewidth=1.5, markersize=5,
            )
        ax.set_title(f"causal Gaussian sigma = {sigma} ms")
        ax.set_xlabel("Session date")
        ax.grid(True, alpha=0.3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
    axes[0].set_ylabel("Mean velocity correlation (best lag per config)")
    axes[-1].legend(loc="best", fontsize=9)
    fig.suptitle("Origin Kording Kalman — sweep across sessions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved trend figure: {out_path}")


def main():
    sessions = list_sessions()
    print(f"Sweeping {len(sessions)} sessions x {len(BIN_SIZES_MS)} bin sizes x "
          f"{len(TARGET_MODES)} target modes")

    import time
    all_records = []
    # Resume mode: if the CSV already exists, skip sessions that are fully present.
    done_sessions = set()
    if CSV_LONG.exists():
        try:
            prev = pd.read_csv(CSV_LONG)
            # consider a session "done" only if it has all (bin, target, sigma) combos
            expected = len(BIN_SIZES_MS) * len(TARGET_MODES) * len(SMOOTH_SIGMAS_MS)
            counts = prev.groupby("session").apply(
                lambda d: d.drop_duplicates(["bin_size_ms", "target_mode", "smooth_sigma_ms"]).shape[0]
            )
            done_sessions = set(counts[counts >= expected].index)
            print(f"[resume] found {len(done_sessions)} completed sessions in {CSV_LONG.name}: "
                  f"{sorted(done_sessions)}")
            all_records.extend(prev.to_dict("records"))
        except Exception as e:
            print(f"[resume] could not parse existing CSV ({e}); starting fresh")
            CSV_LONG.unlink()
            done_sessions = set()
    t_start = time.time()
    for i, session_tag in enumerate(sessions, start=1):
        if session_tag in done_sessions:
            print(f"\n>>> [{i}/{len(sessions)}] {session_tag} -- already in CSV, skipping")
            continue
        session_records = []
        for bin_size_ms in BIN_SIZES_MS:
            lag_ms_grid = list(range(0, LAG_MAX_MS + 1, bin_size_ms))
            for target_mode in TARGET_MODES:
                print(f"\n>>> [{i}/{len(sessions)}] {session_tag} | bin={bin_size_ms}ms | "
                      f"target={target_mode} | {len(lag_ms_grid)} lags")
                try:
                    recs = evaluate_one(
                        session_tag, bin_size_ms, target_mode, lag_ms_grid, SMOOTH_SIGMAS_MS
                    )
                    session_records.extend(recs)
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
        all_records.extend(session_records)
        # Append-mode checkpoint: write header on first session, append after.
        sdf = pd.DataFrame(session_records)
        sdf.to_csv(CSV_LONG, mode="a", header=not CSV_LONG.exists(), index=False)
        elapsed = time.time() - t_start
        est_total = elapsed / i * len(sessions)
        print(f"\n[checkpoint] {i}/{len(sessions)} sessions done | "
              f"{elapsed/60:.1f} min elapsed | est total {est_total/60:.1f} min")

    long_df = pd.DataFrame(all_records)
    print(f"\nWrote {CSV_LONG} ({len(long_df)} rows total)")

    summary_df = summarize(long_df)
    summary_df.to_csv(CSV_SUMMARY, index=False)
    print(f"Wrote {CSV_SUMMARY} ({len(summary_df)} rows)")

    best_df = best_lag_per_config(summary_df)
    print("\n=== Best lag per (session, bin_size, target_mode) ===")
    print(best_df.to_string(index=False))

    plot_trend(best_df, FIG_TREND)


if __name__ == "__main__":
    main()
