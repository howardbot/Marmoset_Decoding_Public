"""Phase 1 of the big parameter sweep: within-day decoding only.

Sweep dimensions (per session):
  outlier_mode       : 0828 trial-41 include vs exclude (no-op for other sessions)
  bin_size_ms        : 10, 20, 30, 40, 50
  trajectory smoother: savgol (default), butter order-2, butter order-4
  target_mode        : relative_position, relative_velocity
  lag_ms             : bin-integer multiples of [0, 150]
  decoder            : kalman (KordingLab), wiener (KordingLab)
  wiener history_ms  : 50, 100 (only when decoder=wiener)
  fold               : 0..4 (5-fold CV by trial)

Output:
  Results/generalization/big_sweep_withinday_long.csv
    One row per (cell × lag × fold × decoder × history_ms).
    Cell = (session, outlier_mode, bin_size_ms, smoother, target_mode).

Checkpointing:
  Rows are appended to the CSV after every cell. Re-running the script skips
  any (session, outlier_mode, bin_size_ms, smoother, target_mode) cell that
  already has at least one row in the CSV. Safe to Ctrl-C and resume.

Constants are at the top of the file. No CLI flags by design -- this is a
research sweep, not a tool.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

import decoder_utils as du
from project_config import (
    EXCLUDE_TRIALS,
    GENERALIZATION_RESULTS_DIR,
    TS_INTERFERENCE_R1,
    TS_INTERFERENCE_R2,
)
from Neural_Decoding.decoders import KalmanFilterRegression, WienerFilterRegression
from Wiener_filter import history_bins_for

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
SESSIONS_R1 = list(TS_INTERFERENCE_R1)
SESSIONS_R2 = list(TS_INTERFERENCE_R2)
ALL_SESSIONS = SESSIONS_R1 + SESSIONS_R2

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
BIN_SIZES_MS = [10, 20, 30, 40, 50]
SMOOTHERS = [
    ("savgol",     {"smoother": "savgol"}),
    ("butter_o2",  {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}),
    ("butter_o4",  {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 4}),
]
TARGET_MODES = ["relative_position", "relative_velocity"]
WIENER_HISTORY_MS = [50, 100]
MAX_LAG_MS = 150
N_FOLDS = 5
SMOOTH_SIGMA_MS = 50         # neural causal Gaussian sigma (fixed)
UNIT_QUALITIES = ("good", "mua")
TRIAL_RESULTS = ("S", "F")
SEED = 0

REPO_ROOT = GENERALIZATION_RESULTS_DIR.parents[1]
OUT_CSV = GENERALIZATION_RESULTS_DIR / "big_sweep_withinday_long.csv"

# Number of worker processes for cell-level parallelism. M2 has 4 perf cores;
# we leave one free for the OS and main process. Each worker also pins BLAS to
# a single thread (see _init_worker) to avoid 3*N oversubscription.
N_WORKERS = 3


def lag_ms_grid(bin_size_ms, max_lag_ms=MAX_LAG_MS):
    """Bin-integer multiples in [0, max_lag_ms]."""
    return list(range(0, max_lag_ms + 1, bin_size_ms))


# ---------------------------------------------------------------------------
# Decoder helpers
# ---------------------------------------------------------------------------
def corr_1d(yt, yp):
    good = np.isfinite(yt) & np.isfinite(yp)
    if good.sum() < 2:
        return np.nan
    yt = np.asarray(yt[good], dtype=float)
    yp = np.asarray(yp[good], dtype=float)
    if np.std(yt) == 0 or np.std(yp) == 0:
        return np.nan
    return float(np.corrcoef(yt, yp)[0, 1])


def m2_per_trial(X_te, pred, meta_te):
    """M2: mean Pearson r across all (trial, dim) cells."""
    vals = []
    for _, idx in meta_te.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 4:
            continue
        for d in range(X_te.shape[1]):
            vals.append(corr_1d(X_te[idx, d], pred[idx, d]))
    return float(np.nanmean(vals)) if vals else np.nan


def fit_kalman(X_tr, Y_tr, X_te, Y_te, meta_te):
    """Center + z-score + KordingLab Kalman; predict trial-by-trial."""
    keep = np.nanstd(Y_tr, axis=0) > 1e-12
    Yk_tr, Yk_te = Y_tr[:, keep], Y_te[:, keep]
    Y_mean = np.nanmean(Yk_tr, axis=0)
    Y_std = np.nanstd(Yk_tr, axis=0)
    X_mean = np.nanmean(X_tr, axis=0)
    Yk_tr = (Yk_tr - Y_mean) / Y_std
    Yk_te = (Yk_te - Y_mean) / Y_std
    X_tr_c = X_tr - X_mean
    X_te_c = X_te - X_mean
    model = KalmanFilterRegression(C=1)
    model.fit(Yk_tr, X_tr_c)
    pred = np.full_like(X_te_c, np.nan, dtype=float)
    for _, idx in meta_te.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        pred[idx] = np.asarray(model.predict(Yk_te[idx], X_te_c[idx]), dtype=float)
    return X_te_c, pred


def fit_wiener(X_tr, Y_tr, meta_tr, X_te, Y_te, meta_te, history_bins):
    """History embedding + mean-centered OLS Wiener; restore mean on predict."""
    F_tr, Xh_tr, _ = du.make_history_features(X_tr, Y_tr, meta_tr, history_bins, lag_bins=0)
    F_te, Xh_te, meta_te_h = du.make_history_features(X_te, Y_te, meta_te, history_bins, lag_bins=0)
    if len(F_tr) == 0 or len(F_te) == 0:
        return None, None, None
    F_mean = np.nanmean(F_tr, axis=0)
    X_mean = np.nanmean(Xh_tr, axis=0)
    model = WienerFilterRegression()
    model.fit(F_tr - F_mean, Xh_tr - X_mean)
    pred = model.predict(F_te - F_mean) + X_mean
    return Xh_te, pred, meta_te_h


def filter_trials(X, Y, meta, exclude_trial_nums):
    if not exclude_trial_nums:
        return X, Y, meta
    mask = ~meta["trial_number"].isin(exclude_trial_nums).to_numpy()
    return X[mask], Y[mask], meta[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
CELL_COLS = ["session", "outlier_mode", "bin_size_ms", "smoother", "target_mode"]


def load_done_cells(csv_path):
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=lambda c: c in CELL_COLS, low_memory=True)
    if not all(c in df.columns for c in CELL_COLS):
        return set()
    return set(map(tuple, df[CELL_COLS].drop_duplicates().to_numpy()))


def append_rows(csv_path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------
def run_cell(session, outlier_mode, bin_size_ms, smoother_label, smoother_kw, target_mode):
    """Run all (lag x fold x decoder x history) configs for one cell."""
    bin_size_s = bin_size_ms / 1000.0

    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_size_s

    io, nwb_prc, reach_tbl = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb_prc, reach_tbl, target_mode,
            bin_size=bin_size_s,
            unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS,
            trial_window="start_to_peak",
            **smoother_kw,
        )
    finally:
        io.close()

    exclude_list = (
        EXCLUDE_TRIALS[session]
        if outlier_mode == "exclude" and session in EXCLUDE_TRIALS
        else []
    )
    X, Y, meta = filter_trials(X, Y, meta, exclude_list)
    if len(X) < 100 or meta["trial_number"].nunique() < N_FOLDS:
        return []

    sigma_bins = SMOOTH_SIGMA_MS / bin_size_ms
    Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=sigma_bins)

    # Pre-compute (train_trial_set, test_trial_set) per fold once.
    # Trial membership is invariant under apply_lag (which only drops the first
    # lag_bins samples of each trial), so we can split by trial_number *after*
    # applying lag and avoid calling apply_lag inside the fold loop.
    fold_trial_sets = []
    for tr_mask, te_mask in du.kfold_split_by_trial(meta, n_splits=N_FOLDS, random_seed=SEED):
        fold_trial_sets.append((
            set(meta.loc[tr_mask, "trial_number"].unique()),
            set(meta.loc[te_mask, "trial_number"].unique()),
        ))

    rows = []
    for lag_ms in lag_ms_grid(bin_size_ms):
        lag_bins = lag_ms // bin_size_ms
        # Apply lag ONCE on the full session, not once per fold.
        X_lag, Y_lag, meta_lag = du.apply_lag(X, Y_sm, meta, lag_bins, verbose=False)
        lag_trials_arr = meta_lag["trial_number"].to_numpy()

        for fold_idx, (tr_trials, te_trials) in enumerate(fold_trial_sets):
            tr_mask_lag = np.isin(lag_trials_arr, list(tr_trials))
            te_mask_lag = np.isin(lag_trials_arr, list(te_trials))
            X_tr_l, Y_tr_l = X_lag[tr_mask_lag], Y_lag[tr_mask_lag]
            X_te_l, Y_te_l = X_lag[te_mask_lag], Y_lag[te_mask_lag]
            meta_tr_l = meta_lag[tr_mask_lag].reset_index(drop=True)
            meta_te_l = meta_lag[te_mask_lag].reset_index(drop=True)
            if len(X_tr_l) < 50 or len(X_te_l) < 20:
                continue

            base = dict(
                session=session,
                outlier_mode=outlier_mode,
                bin_size_ms=bin_size_ms,
                smoother=smoother_label,
                target_mode=target_mode,
                lag_ms=lag_ms,
                fold=fold_idx,
            )

            # Kalman
            try:
                X_te_c, pred = fit_kalman(X_tr_l, Y_tr_l, X_te_l, Y_te_l, meta_te_l)
                m2 = m2_per_trial(X_te_c, pred, meta_te_l)
                rows.append({**base, "decoder": "kalman", "history_ms": np.nan, "M2": m2})
            except Exception as e:
                rows.append({**base, "decoder": "kalman", "history_ms": np.nan,
                             "M2": np.nan, "error": f"{type(e).__name__}:{e}"[:120]})

            # Wiener (history sweep)
            for history_ms in WIENER_HISTORY_MS:
                hb = history_bins_for(history_ms, bin_size_ms)
                try:
                    Xh_te, pred_w, meta_te_h = fit_wiener(
                        X_tr_l, Y_tr_l, meta_tr_l,
                        X_te_l, Y_te_l, meta_te_l, hb,
                    )
                    if Xh_te is None:
                        continue
                    m2 = m2_per_trial(Xh_te, pred_w, meta_te_h)
                    rows.append({**base, "decoder": "wiener", "history_ms": history_ms, "M2": m2})
                except Exception as e:
                    rows.append({**base, "decoder": "wiener", "history_ms": history_ms,
                                 "M2": np.nan, "error": f"{type(e).__name__}:{e}"[:120]})

    return rows


# ---------------------------------------------------------------------------
# Multiprocessing
# ---------------------------------------------------------------------------
def _init_worker():
    """Per-worker setup: pin BLAS to 1 thread so N_WORKERS * BLAS_threads does
    not oversubscribe the M2's 8 cores. Numpy/scipy/sklearn will still use
    vectorised inner loops, just not multi-threaded ones."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    warnings.filterwarnings("ignore")


def _worker_run_cell(args):
    """Top-level wrapper so the function is picklable for ProcessPoolExecutor."""
    session, outlier_mode, bin_size_ms, smoother_label, smoother_kw, target_mode = args
    try:
        rows = run_cell(session, outlier_mode, bin_size_ms, smoother_label, smoother_kw, target_mode)
        return ("ok", args, rows, None)
    except Exception as e:
        return ("fail", args, None, f"{type(e).__name__}:{e}"[:200])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_cells(OUT_CSV)
    print(f"[resume] {len(done)} cells already in {OUT_CSV.name}")

    cells = []
    for session in ALL_SESSIONS:
        outlier_modes = (
            ("include", "exclude") if session in EXCLUDE_TRIALS else ("include",)
        )
        for outlier_mode in outlier_modes:
            for bin_size_ms in BIN_SIZES_MS:
                for smoother_label, smoother_kw in SMOOTHERS:
                    for target_mode in TARGET_MODES:
                        cell = (session, outlier_mode, bin_size_ms, smoother_label, target_mode)
                        if cell in done:
                            continue
                        cells.append((session, outlier_mode, bin_size_ms,
                                      smoother_label, smoother_kw, target_mode))

    print(f"[plan] {len(cells)} cells to compute, {N_WORKERS} workers")
    if not cells:
        print("Nothing to do.")
        return
    total_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as pool:
        futures = {pool.submit(_worker_run_cell, args): args for args in cells}
        for i, fut in enumerate(as_completed(futures), 1):
            status, args, rows, err = fut.result()
            session, outlier_mode, bin_size_ms, smoother_label, _, target_mode = args
            short_session = session.replace("TSAL", "")[:8]
            tag = (f"{short_session} outlier={outlier_mode} bin={bin_size_ms} "
                   f"sm={smoother_label} tgt={target_mode}")
            if status == "ok":
                append_rows(OUT_CSV, rows)
                print(f"[{i}/{len(cells)}] {tag} -> {len(rows)} rows "
                      f"(elapsed {(time.time()-total_start)/60:.1f} min)")
            else:
                print(f"[{i}/{len(cells)}] {tag} FAIL: {err}")
                append_rows(OUT_CSV, [{
                    "session": session, "outlier_mode": outlier_mode,
                    "bin_size_ms": bin_size_ms, "smoother": smoother_label,
                    "target_mode": target_mode, "decoder": "ERROR",
                    "lag_ms": np.nan, "fold": np.nan, "history_ms": np.nan, "M2": np.nan,
                    "error": err,
                }])

    print(f"\nDONE. Wall clock: {(time.time()-total_start)/60:.1f} min")
    print(f"CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
