"""F1: Cross-day decoder transfer matrix (Kalman, locked config) + supplement.

Main figure (fig1_crossday_matrix_kalman.png), two panels:
  Left  : Kalman / relative_velocity (LOCKED_CONFIG primary target)
  Right : Kalman / relative_position (SECONDARY_TARGET, parallel panel)

Each panel is a 15x15 matrix of M2 (mean per-trial Pearson r), with:
  * red frames on the diagonal (within-day 5-fold CV)
  * a thick white split line at the R1 | R2 boundary
  * shared viridis colormap with a common vmax (99th percentile)

Reads ``Results/generalization/big_sweep_crossday_long.csv`` via
``plotting_common.load_sweep``; all configuration deviations from LOCKED_CONFIG
must come through that module so figures stay aligned.

Supplement (defends the matrix diagonal):
  fig1b_example_traj_position.png
  fig1b_example_traj_velocity.png

A colour in the matrix tells a reviewer nothing about what a given ``corr``
*looks like* as a movement. The supplement reconstructs the held-out
(out-of-fold) Kalman predictions for one representative session at LOCKED_CONFIG
and overlays true vs decoded trajectories for 10 trials chosen to span the
per-trial corr distribution (NOT cherry-picked). Each panel is titled with that
trial's corr -- the same quantity averaged into a Fig 1 diagonal cell.

Writes all three PNGs into ``Results/generalization/figures/``.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

import decoder_utils as du
from Neural_Decoding.decoders import KalmanFilterRegression
from plotting_common import (
    SECONDARY_TARGET, CMAP_PRIMARY, METRIC_LABEL, LOCKED_CONFIG,
    config_caption, draw_diagonal_frames, draw_r1r2_split, ensure_fig_dir,
    filter_locked, load_sweep, pivot_matrix, set_session_ticks, shared_vmax,
)

warnings.filterwarnings("ignore")


# ===========================================================================
# Main matrix figure
# ===========================================================================
def plot_panel(ax, mat, title, vmin, vmax):
    im = ax.imshow(mat.values, cmap=CMAP_PRIMARY, vmin=vmin, vmax=vmax, aspect="equal")
    draw_diagonal_frames(ax)
    draw_r1r2_split(ax)
    set_session_ticks(ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Test session")
    ax.set_ylabel("Train session")
    return im


def make_matrix_figure(out_dir):
    df = load_sweep()

    mat_vel = pivot_matrix(filter_locked(df))                                  # vel
    mat_pos = pivot_matrix(filter_locked(df, target_mode=SECONDARY_TARGET))    # pos

    vmax = shared_vmax(mat_vel.values, mat_pos.values, quantile=0.99)
    vmin = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    plot_panel(axes[0], mat_vel, "Kalman · velocity", vmin, vmax)
    im1 = plot_panel(axes[1], mat_pos, "Kalman · position", vmin, vmax)

    cbar = fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(METRIC_LABEL)

    fig.suptitle(
        f"Cross-day decoder transfer  ·  R1 (13 days, static) | R2 (2 days, interference)\n"
        f"{config_caption()}",
        fontsize=11, y=1.02,
    )

    out = out_dir / "fig1_crossday_matrix_kalman.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


# ===========================================================================
# Supplement: true vs decoded example trajectories (defends the diagonal)
# ===========================================================================
# Representative strong within-day R1 session (last R1 day). Override if needed.
TRAJ_SESSION = "TSAL20250813_0830_staticAndStaticFree001"
TRAJ_SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
SMOOTH_SIGMA_MS = 50
N_FOLDS = 5
SEED = 0
N_EXAMPLES = 10
EXCLUDE_TRIALS = {"TSAL20250828_0830_interferenceAndInterferenceFree001": [41]}
AXIS_COLORS = ["#1b9e77", "#d95f02", "#7570b3"]  # x, y, z (color-blind safe)


def _oof_predictions(target_mode):
    """{trial_number: (t_sec, true[T,3], pred[T,3])} via 5-fold OOF Kalman.

    Mirrors big_sweep_phase1_withinday.fit_kalman exactly (same centering /
    z-scoring) but keeps per-trial predictions instead of collapsing to corr.
    The train-set mean is added back so curves are in real relative units.
    """
    bin_size_s = LOCKED_CONFIG["bin_size_ms"] / 1000.0
    du.SESSION = TRAJ_SESSION
    du.PROCESSED_NWB = du.DATA_DIR / f"{TRAJ_SESSION}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_size_s

    io, nwb_prc, reach_tbl = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb_prc, reach_tbl, target_mode,
            bin_size=bin_size_s,
            unit_qualities=("good", "mua"),
            trial_results=("S", "F"),
            trial_window="start_to_peak",
            **TRAJ_SMOOTHER_KW,
        )
    finally:
        io.close()

    excl = EXCLUDE_TRIALS.get(TRAJ_SESSION, [])
    if excl:
        keep = ~meta["trial_number"].isin(excl).to_numpy()
        X, Y, meta = X[keep], Y[keep], meta[keep].reset_index(drop=True)

    sigma_bins = SMOOTH_SIGMA_MS / LOCKED_CONFIG["bin_size_ms"]
    Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=sigma_bins)
    X, Y_sm, meta = du.apply_lag(X, Y_sm, meta, 0, verbose=False)  # lag=0 locked

    results = {}
    for tr_mask, te_mask in du.kfold_split_by_trial(meta, n_splits=N_FOLDS, random_seed=SEED):
        X_tr, Y_tr = X[tr_mask], Y_sm[tr_mask]
        X_te, Y_te = X[te_mask], Y_sm[te_mask]
        meta_te = meta[te_mask].reset_index(drop=True)

        keep = np.nanstd(Y_tr, axis=0) > 1e-12
        Yk_tr, Yk_te = Y_tr[:, keep], Y_te[:, keep]
        Y_mean, Y_std = np.nanmean(Yk_tr, axis=0), np.nanstd(Yk_tr, axis=0)
        X_mean = np.nanmean(X_tr, axis=0)
        Yk_tr = (Yk_tr - Y_mean) / Y_std
        Yk_te = (Yk_te - Y_mean) / Y_std

        model = KalmanFilterRegression(C=1)
        model.fit(Yk_tr, X_tr - X_mean)

        for _, idx in meta_te.groupby("trial_number").indices.items():
            idx = np.asarray(idx)
            tn = int(meta_te.loc[idx[0], "trial_number"])
            pred = np.asarray(model.predict(Yk_te[idx], X_te[idx] - X_mean), dtype=float) + X_mean
            t = np.arange(len(idx)) * bin_size_s
            results[tn] = (t, X_te[idx], pred)
    return results


def _trial_corr(true, pred):
    """Mean Pearson r across the 3 axes for one trial."""
    vals = []
    for d in range(true.shape[1]):
        a, b = true[:, d], pred[:, d]
        good = np.isfinite(a) & np.isfinite(b)
        if good.sum() < 4 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
            continue
        vals.append(np.corrcoef(a[good], b[good])[0, 1])
    return float(np.nanmean(vals)) if vals else np.nan


def _pick_representative(results, n=N_EXAMPLES):
    """Pick n trials spanning the per-trial corr distribution (quantile-spaced)."""
    scored = [(tn, _trial_corr(t[1], t[2])) for tn, t in results.items()]
    scored = [(tn, c) for tn, c in scored if np.isfinite(c) and len(results[tn][0]) >= 4]
    scored.sort(key=lambda x: x[1])
    if len(scored) <= n:
        return scored
    qs = np.linspace(0, len(scored) - 1, n).round().astype(int)
    return [scored[i] for i in qs]


def make_trajectory_figure(out_dir, target_mode, names, ylabel, fname):
    results = _oof_predictions(target_mode)
    picks = _pick_representative(results)
    med = np.nanmedian([c for _, c in picks])

    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
    for ax, (tn, corr) in zip(axes.ravel(), picks):
        t, true, pred = results[tn]
        for d, (nm, col) in enumerate(zip(names, AXIS_COLORS)):
            ax.plot(t, true[:, d], color=col, lw=1.8,
                    label=f"{nm} true" if ax is axes[0, 0] else None)
            ax.plot(t, pred[:, d], color=col, lw=1.4, ls="--",
                    label=f"{nm} decoded" if ax is axes[0, 0] else None)
        ax.set_title(f"trial {tn}   corr={corr:.2f}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel("time from reach start (s)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.999, 0.999))

    short = TRAJ_SESSION.replace("TSAL", "")[:8]
    fig.suptitle(
        f"Fig 1 supplement · true vs decoded {target_mode.replace('relative_', '')}  ·  "
        f"session {short}  ·  held-out (5-fold CV) Kalman\n"
        f"10 trials spanning the corr range (median {med:.2f}); solid = true, "
        f"dashed = decoded, color = axis (x/y/z).   {config_caption(target_mode=target_mode)}",
        fontsize=11, y=1.05,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}  (median corr of shown trials = {med:.3f})")


def main():
    out_dir = ensure_fig_dir()
    make_matrix_figure(out_dir)
    make_trajectory_figure(out_dir, "relative_position", ["rel_x", "rel_y", "rel_z"],
                           "relative position", "fig1b_example_traj_position.png")
    make_trajectory_figure(out_dir, "relative_velocity", ["vel_x", "vel_y", "vel_z"],
                           "relative velocity", "fig1b_example_traj_velocity.png")


if __name__ == "__main__":
    main()
