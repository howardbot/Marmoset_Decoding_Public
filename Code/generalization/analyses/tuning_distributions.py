"""Single-unit tuning: is the encoding REPERTOIRE preserved across R1->R2, or does
the cell-level tuning change? (companion to read-out rotation)

Units are re-sorted each day (no cross-day tracking), so we compare tuning
DISTRIBUTIONS between epochs, not per-neuron shifts. Per unit, fit an encoding
model  rate ~ [position(3), velocity(3)]  (z-scored predictors, ridge), 5-fold
cross-validated by trial:
  - cvR2          : tuning quality / how movement-driven the unit is
  - mod_vel/mod_pos : modulation depth = ||velocity / position weights||
  - PD azimuth    : direction of the velocity weight vector
Pool R1 units (13 sessions) vs R2 units (3 sessions) and compare. Reassociation
predicts a PRESERVED repertoire (similar cvR2 / modulation / PD distributions)
despite the rotated population read-out shown in readout_rotation.py.

Output: printed summary + CSV + figure. NOTE: n(R2)=3 sessions.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS, EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
DT = BIN_MS / 1000.0
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
L2, SEED = 1.0, 0
OUT = _THIS.parents[2] / "Results" / "manifold_geometry"
FIG = OUT / "figures"


def load(session, exclude=()):
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = DT
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, "relative_position", bin_size=DT, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    R = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS)   # firing (bins x units)
    # velocity by within-trial finite difference of position
    V = np.zeros_like(X)
    for _, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) >= 2:
            V[idx] = np.gradient(X[idx], DT, axis=0)
    P = np.hstack([X, V])                                                       # [pos3, vel3]
    P = (P - P.mean(0)) / (P.std(0) + 1e-9)
    return P, R, meta

def cv_r2(P, R, meta):
    """5-fold, trial-held-out R2 for the per-unit linear encoding model.

    P is the kinematic design matrix, shape (n_bins, 6):
        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    R is the smoothed neural response matrix, shape (n_bins, n_units).

    For every unit, the fitted model is:
        firing_rate ~= beta_pos · position + beta_vel · velocity + bias

    The regression is vectorized over units: one ridge solve estimates a separate
    column of weights for every unit. Cross-validation is by whole trials, not by
    individual bins, so temporally adjacent bins from the same trial do not leak
    across train/test folds.
    """
    # Accumulate held-out residual and total sums of squares for each unit across
    # all folds. Each array has one entry per unit.
    ss_res = np.zeros(R.shape[1]); ss_tot = np.zeros(R.shape[1])
    for tr, te in du.kfold_split_by_trial(meta, n_splits=5, random_seed=SEED):
        # Skip pathological folds with too few bins to fit or evaluate a stable
        # encoding model.
        if tr.sum() < 30 or te.sum() < 10:
            continue
        # Add a bias/intercept column to the six z-scored kinematic predictors.
        # A has shape (n_train_bins, 7).
        A = np.hstack([P[tr], np.ones((tr.sum(), 1))])
        # Ridge-regularized linear regression:
        #   minimize ||A W - R_train||^2 + L2 * ||W||^2
        # W has shape (7, n_units). Rows 0:6 are position/velocity weights,
        # and the final row is the bias for each unit.
        W = np.linalg.solve(A.T @ A + L2 * np.eye(A.shape[1]), A.T @ R[tr])
        # Predict held-out firing rates for every unit from held-out kinematics.
        # pred has shape (n_test_bins, n_units).
        pred = np.hstack([P[te], np.ones((te.sum(), 1))]) @ W
        # Residual sum of squares: how much held-out firing variance the model
        # failed to explain, accumulated separately for each unit.
        ss_res += ((R[te] - pred) ** 2).sum(0)
        # Total sum of squares: held-out firing variance around that fold's test
        # mean. This is the baseline error from predicting a constant mean rate.
        ss_tot += ((R[te] - R[te].mean(0)) ** 2).sum(0)
    # Cross-validated R2 per unit. Values near 1 mean the kinematic model predicts
    # held-out firing well; 0 means it is no better than the test-fold mean; values
    # below 0 mean it is worse than the mean baseline.
    return 1 - ss_res / (ss_tot + 1e-12)


def unit_table(session, exclude, epoch):
    P, R, meta = load(session, exclude)
    r2 = cv_r2(P, R, meta)
    A = np.hstack([P, np.ones((len(P), 1))])
    W = np.linalg.solve(A.T @ A + L2 * np.eye(A.shape[1]), A.T @ R)[:-1]        # (6, units)
    mod_pos = np.linalg.norm(W[:3], axis=0); mod_vel = np.linalg.norm(W[3:], axis=0)
    az = np.arctan2(W[4], W[3])                                                 # vel azimuth (wy,wx)
    return pd.DataFrame(dict(session=session[4:12],
                             epoch=epoch,
                             cvR2=r2, mod_pos=mod_pos, mod_vel=mod_vel, vel_az=az))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal", choices=sorted(ANIMAL_SESSIONS), default="TS"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = (
        list(sessions) for sessions in ANIMAL_SESSIONS[args.animal]
    )
    suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
    output_csv = OUT / f"tuning_distributions{suffix}.csv"
    figure_path = FIG / f"fig_tuning_distributions{suffix}.png"
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.concat(
        [
            unit_table(s, EXCLUDE_TRIALS.get(s, []), epoch)
            for epoch, sessions in (
                ("R1", sessions_r1),
                ("R2", sessions_r2),
            )
            for s in sessions
        ],
        ignore_index=True,
    )
    df.insert(0, "animal", args.animal)
    df.to_csv(output_csv, index=False)
    R1, R2 = df[df.epoch == "R1"], df[df.epoch == "R2"]
    print(f"units: R1={len(R1)} ({R1.session.nunique()} sess)  R2={len(R2)} ({R2.session.nunique()} sess)")
    # NOTE: this Mann-Whitney is ACROSS UNITS (the sampling unit), which is more defensible
    # than a pair-level test, but units are pooled across only a few sessions (R2 = 3), so
    # they are not fully independent — read p as approximate, and the conclusion rests on
    # the distributions/medians overlapping rather than on the p-value itself.
    print("\n             R1 median   R2 median   Mann-Whitney p (across-unit, approx.)")
    for col in ["cvR2", "mod_pos", "mod_vel"]:
        p = mannwhitneyu(R1[col], R2[col]).pvalue
        print(f"  {col:8s}   {R1[col].median():8.3f}   {R2[col].median():8.3f}   {p:.4f}")
    thr = 0.05
    print(f"\n  fraction well-tuned (cvR2>{thr}):  R1={ (R1.cvR2>thr).mean()*100:.0f}%   R2={ (R2.cvR2>thr).mean()*100:.0f}%")
    def conc(a):
        return float(np.abs(np.mean(np.exp(1j * a))))
    print(f"  velocity-PD azimuth concentration (0=uniform,1=aligned):  R1={conc(R1.vel_az):.3f}  R2={conc(R2.vel_az):.3f}")
    print("\n  => repertoire PRESERVED if distributions overlap (p>0.05) -> reassociation;"
          "\n     a shifted distribution would mean cell-level encoding also changed.")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    xlabels = ["cross-validated R²  (tuning quality)",
               "velocity modulation depth",
               "position modulation depth"]
    for a, col, ttl, xl in zip(ax, ["cvR2", "mod_vel", "mod_pos"],
                               ["tuning quality (cvR2)", "velocity modulation", "position modulation"],
                               xlabels):
        bins = np.linspace(min(R1[col].quantile(.01), R2[col].quantile(.01)),
                           max(R1[col].quantile(.99), R2[col].quantile(.99)), 30)
        a.hist(R1[col], bins, density=True, alpha=.5, color="#7f8c8d", label="R1")
        a.hist(R2[col], bins, density=True, alpha=.5, color="#e74c3c", label="R2")
        a.set_title(ttl, fontsize=10); a.legend(fontsize=9); a.grid(alpha=.3)
        a.set_xlabel(xl, fontsize=9)
        a.set_ylabel("probability density  (per-unit count, normalised)", fontsize=9)
    fig.suptitle(
        f"{args.animal} single-unit tuning distributions: R1 vs R2 "
        f"(preserved repertoire => reassociation)  n(R2)={len(sessions_r2)}",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {output_csv}\nsaved {figure_path}")


if __name__ == "__main__":
    main()
