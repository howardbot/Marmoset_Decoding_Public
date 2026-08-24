"""Phase 2 of the big parameter sweep: cross-day decoding.

For each (bin_size_ms, target_mode, smoother) outer cell:
  1. Build per-session manifold cache (X, Y_pc, meta, PCA, phase-averaged traj)
     for the 15 sessions in ALL_SESSIONS.
  2. For 0828, build an extra cache with trial 41 removed (the diagnosed
     pose-tracking outlier; see compare_0813_vs_0828.py).
  3. Evaluate every (train, test) pair x lag x decoder x history_ms config.
     Same-day diagonal cells use 5-fold CV in the day's own PC space.
     Off-diagonal cells use CCA-aligned latent canonical space.

Decoders:
  kalman  : KordingLab KalmanFilterRegression on the (canonical) Y_pc
  wiener  : KordingLab WienerFilterRegression on history-embedded canonical Y,
            with mean centering on the training fold (matches Gallego 2020
            "across-day aligned latent decoder").

Output:
  Results/generalization/big_sweep_crossday_long.csv
    One row per (bin, target, smoother, outlier_mode, train_session,
    test_session, lag_ms, decoder, history_ms).

Checkpointing identical to Phase 1: re-running skips any cell already in CSV.

Outlier toggle is only applied to rows where 0828 is the train OR test session.
For all other pairs, outlier_mode is recorded as "include" and run once.

NOTE: When changing the sweep grid here, also update LOCKED_CONFIG in
``plotting_common.py``. The plotting scripts pin to that config, so a sweep
that drops a value LOCKED_CONFIG references will break every figure.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

import decoder_utils as du
from project_config import (
    EXCLUDE_TRIALS,
    GENERALIZATION_RESULTS_DIR,
    INTERFERENCE_SESSIONS,
    TS_INTERFERENCE_R1,
    TS_INTERFERENCE_R2,
    TY_INTERFERENCE_R1,
    TY_INTERFERENCE_R2,
)
from manifold_align import (
    pca_neural, trial_average_pc, cca_align, apply_alignment,
)
from Neural_Decoding.decoders import KalmanFilterRegression, WienerFilterRegression
from Wiener_filter import history_bins_for

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sessions / outliers
# ---------------------------------------------------------------------------
SESSIONS_R1 = list(TS_INTERFERENCE_R1)
SESSIONS_R2 = list(TS_INTERFERENCE_R2)
ALL_SESSIONS = SESSIONS_R1 + SESSIONS_R2
TY_SESSIONS_R1 = list(TY_INTERFERENCE_R1)
TY_SESSIONS_R2 = list(TY_INTERFERENCE_R2)
ANIMAL_SESSIONS = INTERFERENCE_SESSIONS

# ---------------------------------------------------------------------------
# Sweep grid (kept in sync with Phase 1)
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
N_FOLDS_DIAGONAL = 5
SMOOTH_SIGMA_MS = 50           # neural causal Gaussian sigma
K_PCS = 12                     # Gallego-2018 manifold dim (~60% var); v2 re-anchor (was 15)
N_PHASE_BINS = 30
UNIT_QUALITIES = ("good", "mua")
TRIAL_RESULTS = ("S", "F")
SEED = 0

REPO_ROOT = GENERALIZATION_RESULTS_DIR.parents[1]
OUT_CSV = GENERALIZATION_RESULTS_DIR / "big_sweep_crossday_long.csv"

# Outer-combo level parallelism: each worker takes one (bin, smoother, target)
# combo end-to-end (caches + pair evals). The default uses half the logical CPU
# count (8 workers on the current 8-core/16-thread workstation); BLAS is pinned
# to one thread per worker to avoid oversubscription. Each worker holds roughly
# 0.5 GB, so eight workers use about 4 GB for their caches.
N_WORKERS = min(8, max(1, (os.cpu_count() or 2) // 2))
_WORKER_THREAD_LIMITER = None


def lag_ms_grid(bin_size_ms, max_lag_ms=MAX_LAG_MS):
    return list(range(0, max_lag_ms + 1, bin_size_ms))


# ---------------------------------------------------------------------------
# Metric / fit helpers (identical semantics to Phase 1)
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
    vals = []
    for _, idx in meta_te.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 4:
            continue
        for d in range(X_te.shape[1]):
            vals.append(corr_1d(X_te[idx, d], pred[idx, d]))
    return float(np.nanmean(vals)) if vals else np.nan


def kalman_fit_predict(X_tr, Y_tr, X_te, Y_te, meta_te):
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


def wiener_fit_predict(X_tr, Y_tr, meta_tr, X_te, Y_te, meta_te, history_bins):
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
# Cache builder
# ---------------------------------------------------------------------------
def build_cache_entry(session, bin_size_ms, target_mode, smoother_kw,
                     exclude_trial_nums=()):
    """Return {X, Y_pc, meta, PCA_V, PCA_mean, traj} for one (session, config).

    Neural smoothing (causal Gaussian, sigma=50ms) is applied before PCA.
    PCA + phase-averaged trajectory drive the CCA alignment in
    eval_off_diagonal_*.
    """
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

    X, Y, meta = filter_trials(X, Y, meta, exclude_trial_nums)
    sigma_bins = SMOOTH_SIGMA_MS / bin_size_ms
    Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=sigma_bins)
    Y_pc, V, mean = pca_neural(Y_sm, k=K_PCS)
    traj = trial_average_pc(Y_pc, meta, n_phase_bins=N_PHASE_BINS)
    return {
        "session": session, "bin_size_ms": bin_size_ms,
        "X": X, "Y_pc": Y_pc, "meta": meta,
        "PCA_V": V, "PCA_mean": mean, "traj": traj,
        "excluded_trials": list(exclude_trial_nums),
    }


# ---------------------------------------------------------------------------
# Pair evaluation
# ---------------------------------------------------------------------------
def eval_pair(train_cache, test_cache, bin_size_ms):
    """Run all (lag, decoder, history) configs for one (train, test) pair.

    If train_cache == test_cache (same object), use 5-fold CV in Y_pc space
    (no CCA, since CCA with self is identity). Otherwise CCA-align both onto
    canonical Y space and use the full test session as held-out.
    """
    same_session = train_cache is test_cache
    rows_template = []

    if same_session:
        # Diagonal: 5-fold CV in own PC space.
        X = train_cache["X"]
        Y = train_cache["Y_pc"]
        meta = train_cache["meta"]

        # Compute fold trial sets once on the un-lagged meta; trial membership
        # is invariant under apply_lag so the same sets apply to every lag.
        fold_trial_sets = []
        for tr_mask, te_mask in du.kfold_split_by_trial(
            meta, n_splits=N_FOLDS_DIAGONAL, random_seed=SEED,
        ):
            fold_trial_sets.append((
                set(meta.loc[tr_mask, "trial_number"].unique()),
                set(meta.loc[te_mask, "trial_number"].unique()),
            ))

        for lag_ms in lag_ms_grid(bin_size_ms):
            lag_bins = lag_ms // bin_size_ms
            X_lag, Y_lag, meta_lag = du.apply_lag(X, Y, meta, lag_bins, verbose=False)
            lag_trials_arr = meta_lag["trial_number"].to_numpy()
            kalman_scores, wiener_scores = [], {h: [] for h in WIENER_HISTORY_MS}
            for tr_trials, te_trials in fold_trial_sets:
                tr_mask = np.isin(lag_trials_arr, list(tr_trials))
                te_mask = np.isin(lag_trials_arr, list(te_trials))
                X_tr, Y_tr = X_lag[tr_mask], Y_lag[tr_mask]
                X_te, Y_te = X_lag[te_mask], Y_lag[te_mask]
                meta_tr_l = meta_lag[tr_mask].reset_index(drop=True)
                meta_te_l = meta_lag[te_mask].reset_index(drop=True)
                if len(X_tr) < 50 or len(X_te) < 20:
                    continue

                # Kalman
                try:
                    X_te_c, pred = kalman_fit_predict(X_tr, Y_tr, X_te, Y_te, meta_te_l)
                    kalman_scores.append(m2_per_trial(X_te_c, pred, meta_te_l))
                except Exception:
                    kalman_scores.append(np.nan)

                # Wiener per history
                for history_ms in WIENER_HISTORY_MS:
                    hb = history_bins_for(history_ms, bin_size_ms)
                    try:
                        Xh_te, pred_w, meta_te_h = wiener_fit_predict(
                            X_tr, Y_tr, meta_tr_l, X_te, Y_te, meta_te_l, hb,
                        )
                        if Xh_te is None:
                            wiener_scores[history_ms].append(np.nan)
                        else:
                            wiener_scores[history_ms].append(
                                m2_per_trial(Xh_te, pred_w, meta_te_h)
                            )
                    except Exception:
                        wiener_scores[history_ms].append(np.nan)

            rows_template.append({
                "lag_ms": lag_ms, "decoder": "kalman", "history_ms": np.nan,
                "M2_mean": float(np.nanmean(kalman_scores)) if kalman_scores else np.nan,
                "M2_std":  float(np.nanstd(kalman_scores))  if kalman_scores else np.nan,
                "n_folds_used": int(np.sum(np.isfinite(kalman_scores))),
            })
            for history_ms in WIENER_HISTORY_MS:
                sc = wiener_scores[history_ms]
                rows_template.append({
                    "lag_ms": lag_ms, "decoder": "wiener", "history_ms": history_ms,
                    "M2_mean": float(np.nanmean(sc)) if sc else np.nan,
                    "M2_std":  float(np.nanstd(sc))  if sc else np.nan,
                    "n_folds_used": int(np.sum(np.isfinite(sc))),
                })
        return rows_template

    # Off-diagonal: CCA align, full train -> full test.
    W_tr, W_te, m_tr, m_te = cca_align(train_cache["traj"], test_cache["traj"])
    Y_tr_canon = apply_alignment(train_cache["Y_pc"], W_tr, m_tr)
    Y_te_canon = apply_alignment(test_cache["Y_pc"], W_te, m_te)

    for lag_ms in lag_ms_grid(bin_size_ms):
        lag_bins = lag_ms // bin_size_ms
        X_tr_lag, Y_tr_lag, meta_tr_lag = du.apply_lag(
            train_cache["X"], Y_tr_canon, train_cache["meta"], lag_bins, verbose=False,
        )
        X_te_lag, Y_te_lag, meta_te_lag = du.apply_lag(
            test_cache["X"], Y_te_canon, test_cache["meta"], lag_bins, verbose=False,
        )
        if len(X_tr_lag) < 50 or len(X_te_lag) < 20:
            continue

        # Kalman
        try:
            X_te_c, pred = kalman_fit_predict(
                X_tr_lag, Y_tr_lag, X_te_lag, Y_te_lag, meta_te_lag,
            )
            m2_k = m2_per_trial(X_te_c, pred, meta_te_lag)
        except Exception:
            m2_k = np.nan
        rows_template.append({
            "lag_ms": lag_ms, "decoder": "kalman", "history_ms": np.nan,
            "M2_mean": m2_k, "M2_std": np.nan, "n_folds_used": 1,
        })

        # Wiener per history
        for history_ms in WIENER_HISTORY_MS:
            hb = history_bins_for(history_ms, bin_size_ms)
            try:
                Xh_te, pred_w, meta_te_h = wiener_fit_predict(
                    X_tr_lag, Y_tr_lag, meta_tr_lag,
                    X_te_lag, Y_te_lag, meta_te_lag, hb,
                )
                m2_w = m2_per_trial(Xh_te, pred_w, meta_te_h) if Xh_te is not None else np.nan
            except Exception:
                m2_w = np.nan
            rows_template.append({
                "lag_ms": lag_ms, "decoder": "wiener", "history_ms": history_ms,
                "M2_mean": m2_w, "M2_std": np.nan, "n_folds_used": 1,
            })

    return rows_template


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
CELL_COLS = ["bin_size_ms", "smoother", "target_mode", "outlier_mode",
             "train_session", "test_session"]


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
    if not write_header:
        existing_columns = list(pd.read_csv(csv_path, nrows=0).columns)
        missing_columns = [
            column for column in existing_columns if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"cannot append rows missing existing columns: {missing_columns}"
            )
        # Old TS checkpoints predate the optional animal column. Preserve their
        # exact schema so a resumed append cannot shift fields under the header.
        df = df.reindex(columns=existing_columns)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------
def pair_involves_0828(train_session, test_session):
    return any("20250828" in s for s in (train_session, test_session))


def expand_pair_cells(
    bin_size_ms,
    smoother_label,
    target_mode,
    done,
    r1_sessions,
    r2_sessions,
    pair_scope,
):
    """List (train, test, outlier_mode) cells to evaluate for one outer config."""
    cells = []
    if pair_scope == "cross-epoch":
        ordered_pairs = (
            list(product(r1_sessions, r2_sessions))
            + list(product(r2_sessions, r1_sessions))
        )
    elif pair_scope == "all":
        sessions = list(r1_sessions) + list(r2_sessions)
        ordered_pairs = list(product(sessions, repeat=2))
    else:
        raise ValueError(f"unknown pair_scope={pair_scope!r}")

    for train_session, test_session in ordered_pairs:
        outlier_modes = (
            ("include", "exclude") if pair_involves_0828(train_session, test_session)
            else ("include",)
        )
        for outlier_mode in outlier_modes:
            cell = (
                bin_size_ms,
                smoother_label,
                target_mode,
                outlier_mode,
                train_session,
                test_session,
            )
            if cell in done:
                continue
            cells.append((train_session, test_session, outlier_mode))
    return cells


def _init_worker():
    """Pin each worker to one BLAS thread to avoid process oversubscription."""
    global _WORKER_THREAD_LIMITER
    _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    warnings.filterwarnings("ignore")


def run_outer_combo(args):
    """Worker entry point: build caches + evaluate all pending pairs for one
    (bin_size_ms, smoother_label, smoother_kw, target_mode) combination.

    Returns (status, outer_key, rows, err_msg). All needed pair_cells are
    passed in so the worker does not need to know about resume state.
    """
    (
        bin_size_ms,
        smoother_label,
        smoother_kw,
        target_mode,
        pair_cells,
        animal,
    ) = args
    outer_key = (bin_size_ms, smoother_label, target_mode)
    try:
        need_include = set()
        need_exclude = set()
        for train_session, test_session, outlier_mode in pair_cells:
            target_set = need_exclude if outlier_mode == "exclude" else need_include
            target_set.add(train_session)
            target_set.add(test_session)

        cache_include = {
            s: build_cache_entry(s, bin_size_ms, target_mode, smoother_kw, ())
            for s in need_include
        }
        cache_exclude = {
            s: build_cache_entry(s, bin_size_ms, target_mode, smoother_kw,
                                EXCLUDE_TRIALS.get(s, []))
            for s in need_exclude
        }

        rows = []
        for train_session, test_session, outlier_mode in pair_cells:
            tr_cache = (cache_exclude if (outlier_mode == "exclude" and train_session in cache_exclude)
                        else cache_include)[train_session]
            te_cache = (cache_exclude if (outlier_mode == "exclude" and test_session in cache_exclude)
                        else cache_include)[test_session]
            try:
                pair_rows = eval_pair(tr_cache, te_cache, bin_size_ms)
            except Exception as e:
                pair_rows = [{
                    "lag_ms": np.nan, "decoder": "ERROR", "history_ms": np.nan,
                    "M2_mean": np.nan, "M2_std": np.nan, "n_folds_used": 0,
                    "error": f"{type(e).__name__}:{e}"[:200],
                }]
            for r in pair_rows:
                rows.append({
                    "animal": animal,
                    "bin_size_ms": bin_size_ms,
                    "smoother": smoother_label,
                    "target_mode": target_mode,
                    "outlier_mode": outlier_mode,
                    "train_session": train_session,
                    "test_session": test_session,
                    **r,
                })
        return ("ok", outer_key, rows, None)
    except Exception as e:
        return ("fail", outer_key, None, f"{type(e).__name__}:{e}"[:200])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal",
        type=str.upper,
        choices=sorted(ANIMAL_SESSIONS),
        default="TS",
    )
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument(
        "--pair-scope",
        choices=("all", "cross-epoch"),
        default="all",
    )
    parser.add_argument(
        "--bin-sizes",
        nargs="+",
        type=int,
        choices=BIN_SIZES_MS,
        default=list(BIN_SIZES_MS),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGET_MODES,
        default=list(TARGET_MODES),
    )
    parser.add_argument(
        "--smoothers",
        nargs="+",
        choices=[label for label, _ in SMOOTHERS],
        default=[label for label, _ in SMOOTHERS],
    )
    parser.add_argument("--max-pair-cells", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.max_pair_cells is not None and args.max_pair_cells < 1:
        raise ValueError("max-pair-cells must be positive")

    r1_sessions, r2_sessions = ANIMAL_SESSIONS[args.animal]
    output_csv = args.output
    if output_csv is None:
        suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
        output_csv = OUT_CSV.with_name(OUT_CSV.stem + suffix + OUT_CSV.suffix)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_cells(output_csv)
    print(f"[resume] {len(done)} (bin, smoother, target, outlier, train, test) "
          f"cells already in {output_csv.name}")

    smoother_lookup = dict(SMOOTHERS)
    outer_combos = [
        (bs, sl, smoother_lookup[sl], tm)
        for bs in args.bin_sizes
        for sl in args.smoothers
        for tm in args.targets
    ]

    # Filter outer combos to those with pending pair cells.
    pending = []
    total_pending_pairs = 0
    for bin_size_ms, smoother_label, smoother_kw, target_mode in outer_combos:
        pair_cells = expand_pair_cells(
            bin_size_ms,
            smoother_label,
            target_mode,
            done,
            r1_sessions,
            r2_sessions,
            args.pair_scope,
        )
        if args.max_pair_cells is not None:
            pair_cells = pair_cells[:args.max_pair_cells]
        if pair_cells:
            pending.append((
                bin_size_ms,
                smoother_label,
                smoother_kw,
                target_mode,
                pair_cells,
                args.animal,
            ))
            total_pending_pairs += len(pair_cells)

    print(f"[plan] {len(pending)}/{len(outer_combos)} outer combos pending, "
          f"{total_pending_pairs} pair cells total, {args.workers} workers, "
          f"animal={args.animal}, scope={args.pair_scope}")
    if not pending:
        print("Nothing to do.")
        return

    total_start = time.time()
    worker_count = min(args.workers, len(pending))
    with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_worker) as pool:
        futures = {pool.submit(run_outer_combo, task): task for task in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            status, outer_key, rows, err = fut.result()
            bin_size_ms, smoother_label, target_mode = outer_key
            tag = f"bin={bin_size_ms} sm={smoother_label} tgt={target_mode}"
            if status == "ok":
                append_rows(output_csv, rows)
                print(f"[{i}/{len(pending)}] {tag} -> {len(rows)} rows "
                      f"(elapsed {(time.time()-total_start)/60:.1f} min)")
            else:
                print(f"[{i}/{len(pending)}] {tag} FAIL: {err}")

    print(f"\nDONE. Wall clock: {(time.time()-total_start)/60:.1f} min")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    freeze_support()
    main()
