"""Key 'why the asymmetry' controls under the robust config
(single-trial phase-matched CCA, K_PCS=15, decode top d=2), for BOTH targets
(relative_velocity and relative_position). Position shows the larger asymmetry
(F2: forward drops ~35% below the R1->R1 baseline vs ~23% for velocity), so the
mechanism conclusions matter more there.

Per target, three controls:
  1. 2x2 train/test decomposition (R1->R1, R1->R2, R2->R1, R2->R2)
  2. trial-count control (match training N across directions)
  3. coverage control (train R2->R1 on the N most-central R2 trials)

Output: printed summary.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from manifold_align import pca_neural, trial_average_pc
from dimension_sweep import align_full
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS, EXCLUDE_TRIALS, N_PHASE_BINS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
    kalman_fit_predict, m2_per_trial,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 12                         # v2 re-anchor (was 15)
D = 12                         # decode ALL canonical dims (was top-2)
N_REPS = 5
SEED = 0
TARGETS = ["relative_velocity", "relative_position"]
OUT = _THIS.parents[2] / "Results" / "manifold_geometry"

# session loading here,
def load(session, target, exclude=()):
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, target, bin_size=BIN_MS / 1000.0, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    Ysm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS)
    Ypc = pca_neural(Ysm, k=K)[0]
    # Also returning trial_average_pc when in the average mode
    return X, Ypc, meta, trial_average_pc(Ypc, meta, n_phase_bins=N_PHASE_BINS)


def decode(Xtr, Ytr, Xte, Yte, mte, mask=None):
    # If we do have mask, then leave the masked value
    if mask is not None:
        Xtr, Ytr = Xtr[mask], Ytr[mask]
    Xc, pred = kalman_fit_predict(Xtr, Ytr[:, :D], Xte, Yte[:, :D], mte)
    return m2_per_trial(Xc, pred, mte)

# Given trial number, creating mask array
def mask_for(meta, trials):
    return meta["trial_number"].isin(trials).to_numpy()

#
def run_target(target, rng, sessions_r1, sessions_r2, n_reps):
    cache = {
        s: load(s, target, EXCLUDE_TRIALS.get(s, []))
        for s in sessions_r1 + sessions_r2
    }
    # 2 sessions a and b, train and test
    def pdec(a, b):
        Ya, Yb = align_full("single_trial", K, cache[a], cache[b], rng)
        if Ya is None:
            return np.nan
        return decode(cache[a][0], Ya, cache[b][0], Yb, cache[b][2])
    # Calculate the average decode score
    def block(pairs):
        v = [pdec(a, b) for a, b in pairs]
        finite = [x for x in v if np.isfinite(x)]
        return float(np.nanmean(finite)) if finite else np.nan
    # All off-diagonal R1-R1 ordered pairs. Use the FULL set for a stable baseline
    # that scales with n (previously a random 26-pair subsample, which made the
    # R1->R1 baseline rng-dependent and noisier than necessary).
    r1r1 = [
        (a, b) for a, b in product(sessions_r1, sessions_r1) if a != b
    ]
    R1R1 = block(r1r1)
    # All pairs
    R1R2 = block(list(product(sessions_r1, sessions_r2)))
    R2R1 = block(list(product(sessions_r2, sessions_r1)))
    R2R2 = block([
        (a, b) for a, b in product(sessions_r2, sessions_r2) if a != b
    ])
    # Reverse - Forward
    asym = R2R1 - R1R2
    if np.isfinite(R2R2):
        train_side = (R2R1 + R2R2) / 2 - (R1R1 + R1R2) / 2
        test_side = (R1R1 + R2R1) / 2 - (R1R2 + R2R2) / 2
    else:
        train_side = np.nan
        test_side = np.nan

    # controls on R1xR2
    """
    um_f  = unmatched forward: R1 full → R2 full
    um_r  = unmatched reverse: R2 full → R1 full
    mc_r  = matched-count reverse: R2 trial-count-matched subset → R1 full
    cov_r = coverage-restricted reverse: R2 central subset → R1 full
    """
    um_f, um_r, mc_r, cov_r = [], [], [], []
    for r1, r2 in product(sessions_r1, sessions_r2):
        Y1, Y2 = align_full("single_trial", K, cache[r1], cache[r2], rng)
        if Y1 is None:
            continue
        X1, m1 = cache[r1][0], cache[r1][2]
        X2, m2 = cache[r2][0], cache[r2][2]
        t1 = list(m1["trial_number"].unique()); t2 = list(m2["trial_number"].unique())
        # Taking the min of #
        N = min(len(t1), len(t2))
        um_f.append(decode(X1, Y1, X2, Y2, m2))
        um_r.append(decode(X2, Y2, X1, Y1, m1))
        # Picking random N trials from R2, does 5 times and take mean value
        mc_r.append(np.mean([
            decode(
                X2,
                Y2,
                X1,
                Y1,
                m1,
                mask_for(m2, rng.choice(t2, N, False)),
            )
            for _ in range(n_reps)
        ]))
        # Get R2 center canonical neural activity
        gm = Y2.mean(0)
        # Take N nearest trials
        cent = sorted(t2, key=lambda t: np.linalg.norm(Y2[mask_for(m2, [t])].mean(0) - gm))[:N]
        cov_r.append(decode(X2, Y2, X1, Y1, m1, mask_for(m2, cent)))
    um = np.mean(um_r) - np.mean(um_f)
    return dict(R1R1=R1R1, R1R2=R1R2, R2R1=R2R1, R2R2=R2R2, asym=asym,
                train_side=train_side, test_side=test_side,
                um=um, mc=np.mean(mc_r) - np.mean(um_f), cov=np.mean(cov_r) - np.mean(um_f))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal", choices=sorted(ANIMAL_SESSIONS), default="TS"
    )
    parser.add_argument("--reps", type=int, default=N_REPS)
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = (
        list(sessions) for sessions in ANIMAL_SESSIONS[args.animal]
    )
    suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
    output_csv = OUT / f"why_controls_singletrial{suffix}.csv"
    rng = np.random.default_rng(SEED)
    summary_rows = []
    for tgt in TARGETS:
        print(f"\n########## {tgt} (single_trial CCA, d={D}) ##########")
        r = run_target(
            tgt, rng, sessions_r1, sessions_r2, args.reps
        )
        print(f"  2x2:   R1->R1={r['R1R1']:.3f}  R1->R2={r['R1R2']:.3f}  "
              f"R2->R1={r['R2R1']:.3f}  R2->R2={r['R2R2']:.3f}")
        print(f"  forward drop below R1->R1 baseline = {r['R1R2']-r['R1R1']:+.3f}   "
              f"reverse vs baseline = {r['R2R1']-r['R1R1']:+.3f}")
        if np.isfinite(r["R2R2"]):
            print(
                f"  asymmetry = {r['asym']:+.3f}  = "
                f"train-side {r['train_side']:+.3f} + "
                f"test-side {r['test_side']:+.3f}"
            )
        else:
            print(
                f"  asymmetry = {r['asym']:+.3f}; train/test-side "
                "decomposition unavailable because n(R2)=1"
            )
        # The "% retained" ratio is only meaningful when the unmatched asymmetry is
        # large enough; for velocity (um ~= 0.01) it explodes into noise, so guard it.
        pct = lambda x: (f"{x / r['um'] * 100:.0f}%" if abs(r['um']) >= 0.02
                         else "N/A (um≈0, ratio not meaningful)")
        print(f"  controls: unmatched asym={r['um']:+.3f} | trial-count {r['mc']:+.3f} "
              f"({pct(r['mc'])}) | coverage {r['cov']:+.3f} ({pct(r['cov'])})")
        summary_rows.append({
            "animal": args.animal,
            "target": tgt,
            "n_r1": len(sessions_r1),
            "n_r2": len(sessions_r2),
            **r,
            "trial_count_retained": (
                r["mc"] / r["um"] if abs(r["um"]) >= 0.02 else np.nan
            ),
            "coverage_retained": (
                r["cov"] / r["um"] if abs(r["um"]) >= 0.02 else np.nan
            ),
        })
    pd.DataFrame(summary_rows).to_csv(output_csv, index=False)
    print(f"\nsaved {output_csv}")


if __name__ == "__main__":
    main()
