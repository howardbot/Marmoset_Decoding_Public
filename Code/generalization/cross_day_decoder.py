"""Build the 13x13 cross-day decoder transfer matrix for task A round 1.

For each ordered pair (train_day, test_day) of r1 sessions:
  - Off-diagonal: fit Kalman on full train day in CCA-aligned canonical space,
    predict on full test day in the same space, report mean velocity correlation.
  - Diagonal: 5-fold by-trial cross-validation within the session in its own PC
    space (no CCA needed); this gives an honest within-day baseline that
    contains the same overfitting protection as off-diagonal entries (which
    never see test-day kinematics during fitting).

All decoder hyperparameters come from `Results/decoder_baseline_summary.md`:
  bin = 20 ms, target = relative_velocity, sigma = 50 ms causal Gaussian,
  units = good + mua, trial window = start_to_peak, k = 15 PCs.
The lag is read per training session from `Results/kalman_sweep_table.csv`.

Outputs (Results/generalization/):
  - cross_day_corr_long.csv     : one row per (train, test, dim)
  - cross_day_corr_matrix.csv   : 13x13 mean-velocity correlation matrix
  - generalization_summary.csv  : within-day vs mean off-diag per training day

Notes:
  - Sessions are filtered to dates in 20250731..20250813 to avoid accidentally
    pulling in round 2 (or any future) NWB files dropped into Data/.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Neural_Decoding.decoders import KalmanFilterRegression

_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_THIS_DIR))
import decoder_utils as du
from manifold_align import (
    pca_neural,
    trial_average_pc,
    cca_align,
    apply_alignment,
)

warnings.filterwarnings("ignore")

REPO_ROOT = _THIS_DIR.parents[1]
DATA_DIR = REPO_ROOT / "Data"
RESULTS_DIR = REPO_ROOT / "Results" / "generalization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Locked configuration (see Results/decoder_baseline_summary.md)
BIN_SIZE_MS = 20
BIN_SIZE_S = BIN_SIZE_MS / 1000.0
TARGET = "relative_velocity"
SMOOTH_SIGMA_MS = 50
UNIT_QUALITIES = ("good", "mua")
TRIAL_WINDOW = "start_to_peak"
K_PCS = 12                     # Gallego-2018 manifold dim (~60% var); v2 re-anchor (was 15)
N_PHASE_BINS = 30
N_SPLITS_DIAGONAL = 5

# Null baseline: circular-shift X_test by a random offset and re-correlate the
# (unchanged) Kalman predictions against the shifted ground truth. Repeated
# N_NULL_SHIFTS times per pair to build a null distribution per cell.
N_NULL_SHIFTS = 10
NULL_SHIFT_MIN_BINS = 50  # at 20 ms bin = 1 second minimum shift to break local structure
NULL_SHIFT_RNG_SEED = 0

# Session epochs. r1 = pre-interference task A; r2 = post-interference task A
# (three days right after the two-week task B period).
R1_DATE_LO, R1_DATE_HI = 20250731, 20250813
R2_DATE_LO, R2_DATE_HI = 20250828, 20250830

# Master switch: skip the circular-shift null when False (~halves the run time
# and skips writing cross_day_null_long.csv / cross_day_null_matrix.csv).
RUN_NULL = True

# Function to grab the data number
def session_date(tag):
    return tag.split("_")[0].replace("TSAL", "")

# Make the date to be int
def session_epoch(tag):
    """Map a session tag to its epoch label ('r1', 'r2', or None if outside both windows)."""
    try:
        d = int(session_date(tag))
    except ValueError:
        return None
    if R1_DATE_LO <= d <= R1_DATE_HI:
        return "r1"
    if R2_DATE_LO <= d <= R2_DATE_HI:
        return "r2"
    return None

#
def list_sessions():
    """Sorted list of r1 then r2 sessions (epoch order, then date within epoch)."""
    # Creating empty for result
    sessions = []
    # Looking for the processed NWB
    for p in sorted(DATA_DIR.glob("*_processed.nwb")):
        tag = p.name.replace("_processed.nwb", "")
        # Make a tag list
        if session_epoch(tag) is not None:
            sessions.append(tag)
    # Stable sort: r1 first then r2, each by date
    sessions.sort(key=lambda t: (0 if session_epoch(t) == "r1" else 1, int(session_date(t))))
    return sessions

# Load the best lag for this
def load_optimal_lag_per_session():
    """Pull per-date optimal lag at standard config from kalman_sweep_table.csv.

    The sweep table is keyed by date (YYYYMMDD); we return a dict keyed by the
    same YYYYMMDD string so callers can look up via session_date(tag).
    """
    csv = REPO_ROOT / "Results" / "kalman_sweep_table.csv"
    df = pd.read_csv(csv)
    mask = (
        (df.bin_size_ms == BIN_SIZE_MS)
        & (df.target == "vel")
        & (df.smooth_sigma_ms == SMOOTH_SIGMA_MS)
    )
    # The sheet after filter
    sub = df[mask]
    # Construct a dictionary
    lags = {str(d): int(l) for d, l in zip(sub.date, sub.lag_ms)}
    if not lags:
        print(f"[warn] no lag rows matched standard config in {csv}; defaulting to 100 ms")
    return lags


def build_session_cache_entry(session_tag):
    """Build (X, Y_smoothed, meta) and the per-day PCA + averaged PC trajectory."""

    du.SESSION = session_tag
    du.PROCESSED_NWB = DATA_DIR / f"{session_tag}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_SIZE_S

    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, TARGET,
            bin_size=BIN_SIZE_S,
            unit_qualities=UNIT_QUALITIES,
            trial_window=TRIAL_WINDOW,
        )
        # bin size count for kernel
        sigma_bins = SMOOTH_SIGMA_MS / BIN_SIZE_MS
        # Smooth the neural data Y
        Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=sigma_bins)
        # Do PCA
        Y_pc, V, mean = pca_neural(Y_sm, k=K_PCS)
        # Sample all the trajectories in 30 and take average
        traj = trial_average_pc(Y_pc, meta, n_phase_bins=N_PHASE_BINS)
    finally:
        io.close()
    return {
        "session": session_tag,
        "X": X,
        "Y_pc": Y_pc,
        "meta": meta,
        "PCA_V": V,
        "PCA_mean": mean,
        "traj": traj,
    }

# Just a corr calculation
def corr_1d(y_true, y_pred):
    # Find position not NaN/inf
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    # too short then skip
    if good.sum() < 2:
        return np.nan
    # Keep good points and make it to array
    yt = np.asarray(y_true[good], dtype=float)
    yp = np.asarray(y_pred[good], dtype=float)
    # if no change then return
    if np.std(yt) == 0 or np.std(yp) == 0:
        return np.nan
    return float(np.corrcoef(yt, yp)[0, 1])


def compute_metric_set(X_te_c, pred, meta_te):
    """From a single (truth, prediction, meta) triple, compute M1 and M2.

    M1 = concatenated Pearson r per dim, averaged across the 3 vel dims.
         BMI-literature default; preserves trial-to-trial baseline structure.
    M2 = per-trial Pearson r per dim, averaged across trials and dims.
         Primary metric used downstream — honest within-trial dynamics.

    M3 (per-trial demean corr) is identical to M2 by construction because
    Pearson r is shift-invariant, so it is not stored separately.
    """
    # Take the dim out, 3/9 (velo/ vel + acc + pos)
    n_dims = X_te_c.shape[1]
    # M1, keep trials together
    m1_per_dim = [corr_1d(X_te_c[:, d], pred[:, d]) for d in range(n_dims)]
    # M2, calculate corr for each trial each dim
    m2_vals = []
    for _, idx in meta_te.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        # if too fewer bins then skip
        if len(idx) < 4:
            continue
        xt = X_te_c[idx]
        pt = pred[idx]
        for d in range(n_dims):
            m2_vals.append(corr_1d(xt[:, d], pt[:, d]))
    return {
        "corr_concat_per_dim": m1_per_dim,                                  # list of 3
        "corr_concat_mean": float(np.nanmean(m1_per_dim)),
        "corr_per_trial_mean": float(np.nanmean(m2_vals)) if m2_vals else np.nan,
    }

# Do the training and prediction
def kalman_fit_predict(X_train, Y_train, X_test, Y_test, meta_test):
    """Drop low-variance features, z-score, center, fit Kalman, predict trial-by-trial."""
    """X_train: train kinematics, T_train x 3
    Y_train: train neural features, T_train x k
    X_test:  test kinematics, T_test x 3
    Y_test:  test neural features, T_test x k
    meta_test: test trial metadata"""
    # keep the dim has changed
    keep = np.nanstd(Y_train, axis=0) > 1e-12
    Y_train, Y_test = Y_train[:, keep], Y_test[:, keep]
    # Doing Z-score & centered
    Y_mean = np.nanmean(Y_train, axis=0)
    Y_std = np.nanstd(Y_train, axis=0)
    X_mean = np.nanmean(X_train, axis=0)
    Y_train = (Y_train - Y_mean) / Y_std
    Y_test = (Y_test - Y_mean) / Y_std
    X_train = X_train - X_mean
    X_test_c = X_test - X_mean
    # Train
    model = KalmanFilterRegression(C=1)
    model.fit(Y_train, X_train)
    pred = np.full_like(X_test_c, np.nan, dtype=float)
    for _, idx in meta_test.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        pred[idx] = np.asarray(model.predict(Y_test[idx], X_test_c[idx]), dtype=float)
    return X_test_c, pred


def null_corrs(X_te_c, pred, rng):
    """Return mean Pearson r across the 3 velocity dims for N_NULL_SHIFTS
    circular shifts of the ground truth.

    The Kalman predictions are unchanged; we only shift the truth so any non-zero
    correlation has to come from coincidental temporal alignment, not from the
    decoder actually capturing structure.
    """
    # Number of time bins
    n = X_te_c.shape[0]
    # If too short can't do circular shift
    if n <= 2 * NULL_SHIFT_MIN_BINS:
        return [np.nan] * X_te_c.shape[1] * N_NULL_SHIFTS
    # Create a list for each velocity dimension, store null corr
    null_per_dim = [[] for _ in range(X_te_c.shape[1])]
    # Do multiple (10) times circular shift
    for _ in range(N_NULL_SHIFTS):
        # Shift random shift range
        shift = int(rng.integers(NULL_SHIFT_MIN_BINS, n - NULL_SHIFT_MIN_BINS))
        X_shifted = np.roll(X_te_c, shift, axis=0)
        for d in range(X_te_c.shape[1]):
            # Calculate corr
            null_per_dim[d].append(corr_1d(X_shifted[:, d], pred[:, d]))
    return null_per_dim  # list of [d] lists each of length N_NULL_SHIFTS


def eval_off_diagonal(train_data, test_data, lag_bins, rng=None, with_null=True):
    """Train on full train day in canonical space, predict on full test day.

    Returns dict containing M1 (concat per-dim and mean) and M2 (per-trial corr),
    plus the null circular-shift distribution for M1.
    """
    # Do CCA
    W_tr, W_te, m_tr, m_te = cca_align(train_data["traj"], test_data["traj"])
    # Projecting
    Y_tr_canon = apply_alignment(train_data["Y_pc"], W_tr, m_tr)
    Y_te_canon = apply_alignment(test_data["Y_pc"], W_te, m_te)
    # Apply the lag
    X_tr_lag, Y_tr_lag, meta_tr_lag = du.apply_lag(
        train_data["X"], Y_tr_canon, train_data["meta"], lag_bins, verbose=False,
    )
    # Apply Lag
    X_te_lag, Y_te_lag, meta_te_lag = du.apply_lag(
        test_data["X"], Y_te_canon, test_data["meta"], lag_bins, verbose=False,
    )
    # Trian & Predict
    X_te_c, pred = kalman_fit_predict(
        X_tr_lag, Y_tr_lag, X_te_lag, Y_te_lag, meta_te_lag
    )
    # Calculate corr performance
    metrics = compute_metric_set(X_te_c, pred, meta_te_lag)
    if with_null:
        null = null_corrs(X_te_c, pred, rng if rng is not None else np.random.default_rng(0))
    else:
        null = [[] for _ in range(X_te_c.shape[1])]
    metrics["null_concat_per_dim"] = null
    return metrics

# With in Day
def eval_diagonal(data, lag_bins, n_splits=N_SPLITS_DIAGONAL, rng=None, with_null=True):
    """5-fold by-trial CV within one session, in its own PC space.

    Returns dict averaged across folds with the same metric keys as eval_off_diagonal.
    """
    X_lag, Y_lag, meta_lag = du.apply_lag(
        data["X"], data["Y_pc"], data["meta"], lag_bins, verbose=False
    )
    fold_metric_sets = []
    null_per_dim = [[] for _ in range(X_lag.shape[1])]
    if rng is None:
        rng = np.random.default_rng(0)
    for train_mask, test_mask in du.kfold_split_by_trial(meta_lag, n_splits=n_splits):
        X_tr, Y_tr = X_lag[train_mask], Y_lag[train_mask]
        X_te, Y_te = X_lag[test_mask], Y_lag[test_mask]
        meta_te = meta_lag[test_mask].reset_index(drop=True)
        X_te_c, pred = kalman_fit_predict(X_tr, Y_tr, X_te, Y_te, meta_te)
        fold_metric_sets.append(compute_metric_set(X_te_c, pred, meta_te))
        if with_null:
            null_fold = null_corrs(X_te_c, pred, rng)
            for d in range(X_te_c.shape[1]):
                null_per_dim[d].extend(null_fold[d])

    # average across folds
    def _avg(key):
        vals = [m[key] for m in fold_metric_sets]
        if isinstance(vals[0], list):
            return list(np.nanmean(np.vstack(vals), axis=0))
        return float(np.nanmean(vals))

    return {
        "corr_concat_per_dim": _avg("corr_concat_per_dim"),
        "corr_concat_mean":    _avg("corr_concat_mean"),
        "corr_per_trial_mean": _avg("corr_per_trial_mean"),
        "null_concat_per_dim": null_per_dim,
    }


def main():
    import time

    sessions = list_sessions()
    epochs = {s: session_epoch(s) for s in sessions}
    n_r1 = sum(1 for s in sessions if epochs[s] == "r1")
    n_r2 = sum(1 for s in sessions if epochs[s] == "r2")
    print(f"Found {n_r1} r1 + {n_r2} r2 = {len(sessions)} sessions total")
    print(f"Null baseline: {'ON' if RUN_NULL else 'OFF'}\n")

    t0 = time.time()
    cache = {}
    failed = []
    for i, s in enumerate(sessions, 1):
        print(f"[cache {i}/{len(sessions)}] building {s}")
        try:
            cache[s] = build_session_cache_entry(s)
        except Exception as e:
            print(f"  SKIP {s}: {type(e).__name__}: {e}")
            failed.append(s)
    sessions = [s for s in sessions if s not in failed]
    if failed:
        print(f"\nSkipped {len(failed)} session(s) due to missing data:")
        for s in failed:
            print(f"  - {s}")
    print(f"Cache built for {len(sessions)} session(s) in {time.time() - t0:.1f}s\n")

    lags = load_optimal_lag_per_session()
    # r2 dates won't be in the lag sweep table; fall back to median r1 optimal lag
    default_lag_ms = int(np.median(list(lags.values()))) if lags else 100
    print(f"Default lag for sessions missing from sweep table: {default_lag_ms} ms\n")

    records = []
    null_records = []
    n = len(sessions)
    master_rng = np.random.default_rng(NULL_SHIFT_RNG_SEED)
    for i, train_s in enumerate(sessions):
        train_date = session_date(train_s)
        train_epoch = epochs[train_s]
        lag_ms = int(lags.get(train_date, default_lag_ms))
        lag_bins = lag_ms // BIN_SIZE_MS
        for j, test_s in enumerate(sessions):
            test_date = session_date(test_s)
            test_epoch = epochs[test_s]
            print(f"[{i*n+j+1}/{n*n}] {train_epoch} {train_date} -> {test_epoch} {test_date} lag={lag_ms}ms")
            try:
                if i == j:
                    out = eval_diagonal(cache[train_s], lag_bins, rng=master_rng, with_null=RUN_NULL)
                else:
                    out = eval_off_diagonal(
                        cache[train_s], cache[test_s], lag_bins,
                        rng=master_rng, with_null=RUN_NULL,
                    )
                per_dim = out["corr_concat_per_dim"]
                corr_concat_mean = out["corr_concat_mean"]
                corr_per_trial_mean = out["corr_per_trial_mean"]
                null = out["null_concat_per_dim"]
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                per_dim = [np.nan, np.nan, np.nan]
                corr_concat_mean = np.nan
                corr_per_trial_mean = np.nan
                null = [[] for _ in range(3)]
            for d, name in enumerate(["vel_x", "vel_y", "vel_z"]):
                records.append({
                    "train_date": train_date,
                    "test_date": test_date,
                    "train_epoch": train_epoch,
                    "test_epoch": test_epoch,
                    "train_session": train_s,
                    "test_session": test_s,
                    "lag_ms": lag_ms,
                    "dim": name,
                    "corr": per_dim[d],                                # back-compat: old "corr" column = concat per dim
                    "corr_concat_mean": corr_concat_mean,              # M1: concatenated Pearson r averaged across 3 dims
                    "corr_per_trial_mean": corr_per_trial_mean,        # M2 (primary): per-trial Pearson r averaged
                })
                for shift_idx, nv in enumerate(null[d]):
                    null_records.append({
                        "train_date": train_date,
                        "test_date": test_date,
                        "train_epoch": train_epoch,
                        "test_epoch": test_epoch,
                        "dim": name,
                        "shift_idx": shift_idx,
                        "null_corr": nv,
                    })

    long_df = pd.DataFrame(records)
    long_df.to_csv(RESULTS_DIR / "cross_day_corr_long.csv", index=False)
    if RUN_NULL and null_records:
        null_df = pd.DataFrame(null_records)
        null_df.to_csv(RESULTS_DIR / "cross_day_null_long.csv", index=False)
        null_mean = null_df.groupby(["train_date", "test_date"])["null_corr"].mean().reset_index()
        null_matrix = null_mean.pivot(index="train_date", columns="test_date", values="null_corr")
        null_matrix.to_csv(RESULTS_DIR / "cross_day_null_matrix.csv")
        print("\n=== Null mean velocity corr matrix (circular-shift baseline) ===")
        print(null_matrix.round(3).to_string())
    else:
        null_matrix = None

    # Per-pair tables — one matrix per metric. Each row's three vel-dim entries are
    # already collapsed inside compute_metric_set, so we take .first() to pull
    # the one constant value per (train, test) pair.
    pair_metrics = long_df.groupby(["train_date", "test_date"]).first().reset_index()
    matrix_M1 = pair_metrics.pivot(index="train_date", columns="test_date", values="corr_concat_mean")
    matrix_M2 = pair_metrics.pivot(index="train_date", columns="test_date", values="corr_per_trial_mean")
    matrix_M1.to_csv(RESULTS_DIR / "cross_day_corr_matrix.csv")
    matrix_M2.to_csv(RESULTS_DIR / "cross_day_corr_per_trial_matrix.csv")
    matrix = matrix_M2  # downstream "primary" matrix is M2

    print("\n=== Cross-day matrix: M2 per-trial corr (primary metric) ===")
    print(matrix_M2.round(3).to_string())
    print("\n=== Cross-day matrix: M1 concat corr (back-compat) ===")
    print(matrix_M1.round(3).to_string())

    # Build epoch-level summary (within-epoch vs cross-epoch corr blocks)
    long_with_epoch = long_df.copy()
    block_means = (
        long_with_epoch.groupby(["train_epoch", "test_epoch"])["corr"]
        .mean()
        .reset_index(name="mean_corr")
    )
    print("\n=== Block means by (train_epoch, test_epoch) ===")
    print(block_means.round(3).to_string(index=False))
    block_means.to_csv(RESULTS_DIR / "epoch_block_means.csv", index=False)

    summary = []
    for s in sessions:
        date = session_date(s)
        ep = epochs[s]
        diag = matrix.loc[date, date]
        off = matrix.loc[date, matrix.columns != date].mean()
        row = {
            "train_date": date,
            "train_epoch": ep,
            "within_day": diag,
            "mean_off_diag": off,
            "drop_within_minus_off": diag - off,
        }
        if null_matrix is not None:
            null_off = null_matrix.loc[date, null_matrix.columns != date].mean()
            row["mean_off_diag_null"] = null_off
            row["real_minus_null_off"] = off - null_off
        summary.append(row)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(RESULTS_DIR / "generalization_summary.csv", index=False)
    print("\n=== Generalization summary ===")
    print(sdf.round(3).to_string(index=False))
    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
