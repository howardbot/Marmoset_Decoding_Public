from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # Code/ for local imports
from decoder_utils import (
    apply_lag, build_decoder_dataset, kfold_split_by_trial,
    load_nwb_and_reach, state_names,
)


# Matched to Origin_Kording_Kalman_filter.py for a direct comparison.
TARGET_MODE = "relative_position_velocity_acceleration"
UNIT_QUALITIES = ("good", "mua")
TRIAL_WINDOW = "start_to_peak"
N_SPLITS = 5
LAG_BINS_LIST = list(range(1, 16))  # 10..150 ms in 10 ms steps
C_VALUE = 1.0  # match Origin's KalmanFilterRegression(C=1)
PINV_RCOND = 1e-10
LAG_GRID_PLOT_PATH = Path(__file__).resolve().parent.parent / "Results" / "trial_aware_kalman_lag_grid.png"


# Kording-style Kalman decoder with trial-aware state transitions.
# neural_* is spike count data, and state_* is the movement state.
class KalmanFilterDecoder:
    def __init__(self, C=1.0, rcond=1e-10):
        # C scales the state noise covariance W, following the Kording implementation.
        self.C = C
        # rcond is used by pseudo-inverse to avoid unstable matrix inversion.
        self.rcond = rcond
        self.model = None

    def fit(self, neural_train, state_train, meta_train):
        # Use column-major matrices to match the original Kording notation:
        # X is state_dim x time, Z is neuron_dim x time.
        X = np.asmatrix(np.asarray(state_train, dtype=float).T)
        Z = np.asmatrix(np.asarray(neural_train, dtype=float).T)
        # Take quantity of time bins
        nt = X.shape[1]

        # Estimate A and W only from within-trial transitions.
        # This avoids connecting the end of one reach to the start of another reach.
        x1_blocks = []
        x2_blocks = []
        # for each trial
        for _, idx in meta_train.groupby("trial_number").indices.items():
            # bin array into np array
            idx = np.asarray(idx)
            # No transition here
            if len(idx) < 2:
                continue
            # Take out trajectory
            X_trial = X[:, idx]
            # Take out all expect the last one
            x1_blocks.append(X_trial[:, :-1])
            # Take out all expect the first one
            x2_blocks.append(X_trial[:, 1:])

        if not x1_blocks:
            raise ValueError("No within-trial transitions available for Kalman fit.")
        # Combine them together
        X1 = np.hstack(x1_blocks)
        X2 = np.hstack(x2_blocks)
        n_transitions = X1.shape[1]

        # State transition model:
        # X[t+1] = A * X[t] + state_noise.
        A = X2 * X1.T * np.linalg.pinv(X1 * X1.T, rcond=self.rcond)
        W = (X2 - A * X1) * (X2 - A * X1).T / n_transitions / self.C

        # Observation model:
        # Z[t] = H * X[t] + observation_noise.
        H = Z * X.T * np.linalg.pinv(X * X.T, rcond=self.rcond)
        Q = (Z - H * X) * (Z - H * X).T / nt

        self.model = [A, W, H, Q]
        return self

    def predict(self, neural_test, state_test, meta_test):
        if self.model is None:
            raise RuntimeError("Model has not been fit.")

        A, W, H, Q = self.model
        # Convert test data to the same Kording matrix layout.
        Z_all = np.asmatrix(np.asarray(neural_test, dtype=float).T)
        X_all = np.asmatrix(np.asarray(state_test, dtype=float).T)
        states_out = np.full(np.asarray(state_test).shape, np.nan, dtype=float)
        num_states = X_all.shape[0]
        eye = np.asmatrix(np.eye(num_states))

        for _, idx in meta_test.groupby("trial_number").indices.items():
            idx = np.asarray(idx)
            if len(idx) == 0:
                continue

            # Reset the Kalman state at the beginning of each held-out trial.
            # The first true state is used only for initialization.
            X = X_all[:, idx]
            Z = Z_all[:, idx]
            states = np.empty(X.shape)
            P = np.asmatrix(np.zeros((num_states, num_states)))
            state = X[:, 0]
            states[:, 0] = np.copy(np.squeeze(state))

            for t in range(X.shape[1] - 1):
                # Prediction step: propagate previous state and uncertainty.
                P_m = A * P * A.T + W
                state_m = A * state
                # Update step: correct the prediction using the next neural observation.
                K = P_m * H.T * np.linalg.pinv(H * P_m * H.T + Q, rcond=self.rcond)
                P = (eye - K * H) * P_m
                state = state_m + K * (Z[:, t + 1] - H * state_m)
                states[:, t + 1] = np.squeeze(state)

            states_out[idx] = np.asarray(states.T)

        return states_out


def corr_score_1d(y_true, y_pred):
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if good.sum() < 2:
        return np.nan
    y_true = np.asarray(y_true[good], dtype=float)
    y_pred = np.asarray(y_pred[good], dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def main():
    io, nwb_prc, reach_tbl = load_nwb_and_reach()
    try:
        X, Y, meta = build_decoder_dataset(
            nwb_prc,
            reach_tbl,
            TARGET_MODE,
            unit_qualities=UNIT_QUALITIES,
            trial_window=TRIAL_WINDOW,
        )
        names = state_names(TARGET_MODE)
        rng_plot = np.random.RandomState(0)

        lag_summary = []
        plot_snapshots = []

        for lag_bins in LAG_BINS_LIST:
            lag_ms = int(round(lag_bins * 10))
            X_lag, Y_lag, meta_lag = apply_lag(X, Y, meta, lag_bins)

            fold_corrs = []
            saved_for_plot = rng_plot.randint(0, N_SPLITS)
            snapshot = None

            print(f"\n--- Lag = {lag_ms} ms (lag_bins={lag_bins}) ---")
            for fold_idx, (train_mask, test_mask) in enumerate(
                kfold_split_by_trial(meta_lag, n_splits=N_SPLITS)
            ):
                X_train, Y_train = X_lag[train_mask], Y_lag[train_mask]
                X_test,  Y_test  = X_lag[test_mask],  Y_lag[test_mask]
                meta_train = meta_lag[train_mask].reset_index(drop=True)
                meta_test  = meta_lag[test_mask].reset_index(drop=True)

                # Identical preprocessing to Origin script for a fair comparison.
                keep = np.nanstd(Y_train, axis=0) > 1e-12
                Y_train, Y_test = Y_train[:, keep], Y_test[:, keep]
                Y_mean, Y_std = np.nanmean(Y_train, axis=0), np.nanstd(Y_train, axis=0)
                X_mean = np.nanmean(X_train, axis=0)
                Y_train = (Y_train - Y_mean) / Y_std
                Y_test  = (Y_test  - Y_mean) / Y_std
                X_train = X_train - X_mean
                X_test  = X_test  - X_mean

                model = KalmanFilterDecoder(C=C_VALUE, rcond=PINV_RCOND)
                model.fit(Y_train, X_train, meta_train)
                pred = model.predict(Y_test, X_test, meta_test)

                corrs = np.asarray([
                    corr_score_1d(X_test[:, i], pred[:, i])
                    for i in range(X_test.shape[1])
                ])
                fold_corrs.append(corrs)
                print(f"Fold {fold_idx}: mean_corr={np.nanmean(corrs):.4f} | "
                      f"per-dim={dict(zip(names, np.round(corrs,3)))}")

                if fold_idx == saved_for_plot:
                    snapshot = (lag_ms, X_test.copy(), pred.copy(), meta_test.copy(), X_mean.copy())

            fold_corrs = np.vstack(fold_corrs)
            mean_per_dim = np.nanmean(fold_corrs, axis=0)
            std_per_dim  = np.nanstd(fold_corrs, axis=0)
            overall_mean = float(np.nanmean(mean_per_dim))
            overall_std  = float(np.nanmean(std_per_dim))
            lag_summary.append((lag_ms, overall_mean, overall_std, mean_per_dim))
            plot_snapshots.append(snapshot)

        print("\n=== Lag sweep summary (overall mean_corr across folds and dims) ===")
        print(f"{'lag_ms':>8} {'mean':>10} {'std':>10}")
        for lag_ms, m, s, _ in lag_summary:
            print(f"{lag_ms:>8d} {m:>+10.4f} {s:>10.4f}")
        best = max(lag_summary, key=lambda r: r[1])
        print(f"\nBest lag = {best[0]} ms  (mean_corr = {best[1]:+.4f})")

        # Per-dim breakdown at best lag
        best_per_dim = best[3]
        print("\nPer-dim mean corr at best lag:")
        for name, val in zip(names, best_per_dim):
            print(f"  {name:>8s} = {val:+.4f}")

        # Plot grid (same as Origin)
        n = len(plot_snapshots)
        ncols = 5
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.5*nrows), sharey=True)
        axes = np.atleast_2d(axes).ravel()
        for ax, snap in zip(axes, plot_snapshots):
            lag_ms, X_test, pred, meta_test, X_mean = snap
            trial_lengths = meta_test.groupby("trial_number").size()
            trial_number = trial_lengths.idxmax()
            idx = np.asarray(meta_test.groupby("trial_number").indices[trial_number])
            t = np.arange(len(idx))
            ax.plot(t, X_test[idx, 1] + X_mean[1], "b", label="true vel_y", linewidth=1)
            ax.plot(t, pred[idx, 1] + X_mean[1], "r", label="pred vel_y", linewidth=1)
            ax.set_title(f"lag={lag_ms}ms  trial={trial_number}", fontsize=9)
            ax.set_xlabel("bin")
        for ax in axes[n:]:
            ax.axis("off")
        axes[0].legend(fontsize=8, loc="upper right")
        fig.suptitle("Trial-aware Kalman | random fold per lag (vel_y)")
        fig.tight_layout()
        LAG_GRID_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(LAG_GRID_PLOT_PATH, dpi=150)
        print(f"Saved lag-grid plot: {LAG_GRID_PLOT_PATH}")
    finally:
        io.close()


if __name__ == "__main__":
    main()
