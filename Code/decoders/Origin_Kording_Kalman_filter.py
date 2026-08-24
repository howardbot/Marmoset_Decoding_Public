from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from Neural_Decoding.decoders import KalmanFilterRegression

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # Code/ for local imports
from decoder_utils import build_decoder_dataset, load_nwb_and_reach, state_names, kfold_split_by_trial, apply_lag

TARGET_MODE = "relative_velocity"
UNIT_QUALITIES = ("good", "mua")
TRIAL_WINDOW = "start_to_peak"
N_SPLITS = 5
LAG_BINS_LIST = list(range(1, 16))  # 10ms..150ms with 10ms step (bin_size=0.01s)
LAG_GRID_PLOT_PATH = Path(__file__).resolve().parents[2] / "Results" / "origin_kording_kalman_lag_grid_1.png"


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
        rng_plot = np.random.RandomState(0)  # each lag pick random for subplot

        lag_summary = []  # (lag_ms, mean_corr, std_corr, per_dim_mean)
        plot_snapshots = []  # each lag  (lag_ms, X_test, pred, meta_test, X_mean)

        for lag_bins in LAG_BINS_LIST:
            lag_ms = int(round(lag_bins * 10))  # bin_size=0.01s
            X_lag, Y_lag, meta_lag = apply_lag(X, Y, meta, lag_bins)

            fold_corrs = []
            saved_for_plot = rng_plot.randint(0, N_SPLITS)  # This lag save for where, subplot
            snapshot = None

            print(f"\n--- Lag = {lag_ms} ms (lag_bins={lag_bins}) ---")
            for fold_idx, (train_mask, test_mask) in enumerate(
                kfold_split_by_trial(meta_lag, n_splits=N_SPLITS)
            ):
                X_train, Y_train = X_lag[train_mask], Y_lag[train_mask]
                X_test,  Y_test  = X_lag[test_mask],  Y_lag[test_mask]
                meta_test = meta_lag[test_mask].reset_index(drop=True)

                # Kording example preprocessing:
                # In this script, Y is the neural input matrix and X is the kinematic output.
                # First remove units with no training-set variance, because they cannot be z-scored.
                # Get rid of the no info units
                keep = np.nanstd(Y_train, axis=0) > 1e-12
                Y_train, Y_test = Y_train[:, keep], Y_test[:, keep]

                # Compute all preprocessing parameters from the training set only.
                # Then apply the same neural mean/std and kinematic mean to the held-out test set.
                Y_mean, Y_std = np.nanmean(Y_train, axis=0), np.nanstd(Y_train, axis=0)
                X_mean = np.nanmean(X_train, axis=0)

                # Z-score the neural inputs, matching the Kording example's "X inputs" step.
                Y_train, Y_test = (Y_train - Y_mean) / Y_std, (Y_test - Y_mean) / Y_std

                # Zero-center the kinematic outputs, matching the Kording example's "y outputs" step.
                X_train, X_test = X_train - X_mean, X_test - X_mean
                #Model
                model = KalmanFilterRegression(C=1)
                model.fit(Y_train, X_train)

                # Allocate the prediction array up front. Filled with NaN so that any bin we
                # somehow miss (shouldn't happen, but safety net) stays NaN instead of 0,
                # which would silently corrupt the correlation downstream.
                pred = np.full_like(X_test, np.nan, dtype=float)
                # Kalman is a stateful filter: its internal state must be re-initialized at
                # each reach onset, otherwise the filter would carry posterior state across
                # unrelated trials. We therefore group test samples by trial_number and run
                # model.predict separately on each trial's contiguous bins.
                for _, idx in meta_test.groupby("trial_number").indices.items():
                    idx = np.asarray(idx)
                    # model.predict here takes (Y_obs, X_init): the second arg is the true
                    # kinematic sequence used only to seed the first-state prior; the filter
                    # then evolves on its own using Y_obs.
                    pred[idx] = np.asarray(model.predict(Y_test[idx], X_test[idx]), dtype=float)

                # Per-dimension Pearson correlation between true and predicted kinematics,
                # computed over ALL test bins concatenated (not per-trial averaged), so the
                # score reflects the joint variance the decoder explains across the held-out
                # trials. corr is scale/shift invariant, robust to the X_mean centering above.
                corrs = np.asarray([corr_score_1d(X_test[:, i], pred[:, i]) for i in range(X_test.shape[1])])
                fold_corrs.append(corrs)
                print(f"Fold {fold_idx}: mean_corr={np.nanmean(corrs):.4f} | per-dim={dict(zip(names, np.round(corrs,3)))}")

                # For each lag we pre-chose ONE fold (saved_for_plot) whose results we will
                # later plot. We copy the arrays because the outer loop variables get
                # rebound on the next iteration, which would otherwise mutate the snapshot.
                if fold_idx == saved_for_plot:
                    snapshot = (lag_ms, X_test.copy(), pred.copy(), meta_test.copy(), X_mean.copy())

            # ---- Aggregate this lag's 5 folds into a single summary row ----
            # Stack 5 fold vectors of length n_dims into (5, n_dims) for easy axis-0 stats.
            fold_corrs = np.vstack(fold_corrs)
            # mean_per_dim: average across folds for each output dimension -> stability check
            #               per dim (e.g., velocity decodes better than position).
            mean_per_dim = np.nanmean(fold_corrs, axis=0)
            # std_per_dim: across-fold spread -> tells us how reliable the per-dim estimate
            #              is. Large std means corr is sensitive to which trials are held out.
            std_per_dim  = np.nanstd(fold_corrs, axis=0)
            # overall_mean: single scalar summarising this lag's decoding quality, used by
            #               the final lag-sweep ranking to pick the best lag.
            overall_mean = float(np.nanmean(mean_per_dim))
            # overall_std: averaged across dims (NOT a true CV std), just a rough scale of
            #              variability to print next to overall_mean.
            overall_std  = float(np.nanmean(std_per_dim))
            # Keep mean_per_dim as the 4th tuple element so we can later dig into which
            # specific dim drove the best lag (without re-running the sweep).
            lag_summary.append((lag_ms, overall_mean, overall_std, mean_per_dim))
            # Stash the held-out plotting snapshot for this lag (one entry per lag, in order).
            plot_snapshots.append(snapshot)

        # ===== Lag summary=====
        print("\n=== Lag sweep summary (overall mean_corr across folds and dims) ===")
        print(f"{'lag_ms':>8} {'mean':>10} {'std':>10}")
        for lag_ms, m, s, _ in lag_summary:
            print(f"{lag_ms:>8d} {m:>+10.4f} {s:>10.4f}")
        best = max(lag_summary, key=lambda r: r[1])
        print(f"\nBest lag = {best[0]} ms  (mean_corr = {best[1]:+.4f})")

        # ===== 15 subplots，each lag random fold longest test trial vel_y =====
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
        fig.suptitle("Original Kording Kalman | random fold per lag (vel_y)")
        fig.tight_layout()
        LAG_GRID_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(LAG_GRID_PLOT_PATH, dpi=150)
        print(f"Saved lag-grid plot: {LAG_GRID_PLOT_PATH}")
        plt.show()
    finally:
        io.close()


if __name__ == "__main__":
    main()
