"""Why is R2 a better cross-day trainer? Control for training trial count.

The asymmetry (R2->R1 > R1->R2) is ~77% a TRAIN-side effect: R2-trained decoders
generalize better. One mundane reason: R2 sessions have more trials (90/95) than
R1 (~75), and more training trials -> better-fit decoder -> better transfer.

Test: for each (R1 day, R2 day) pair, train BOTH directions on the SAME number
of trials N = min(n_R1, n_R2) (subsample the larger side), averaged over random
subsamples. Compare the matched asymmetry to the unmatched (full-data) one.
  * matched asymmetry << unmatched  -> trial count drove it (mundane)
  * matched asymmetry ~ unmatched   -> not trial count (something representational)

Alignment: averaged-trajectory CCA on full data (held fixed); only the decoder's
training trial count is varied, isolating the training-set-size effect.
Config: locked (bin=30, butter_o2, sigma=50ms, K_PCS=15, 0828 trial-41 excluded),
velocity, decode in full canonical space.

Output: printed table + Results/manifold_geometry/r2_trialcount_control.csv
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
from manifold_align import pca_neural, trial_average_pc, cca_align, apply_alignment
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS, EXCLUDE_TRIALS, N_PHASE_BINS, K_PCS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
    kalman_fit_predict, m2_per_trial,
)

warnings.filterwarnings("ignore")

BIN_SIZE_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
TARGET = "relative_velocity"
N_REPS = 5
SEED = 0
REPO_ROOT = _THIS.parents[2]
OUT_CSV = REPO_ROOT / "Results" / "manifold_geometry" / "r2_trialcount_control.csv"


def load(session, exclude=()):
    bin_s = BIN_SIZE_MS / 1000.0
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_s
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, TARGET, bin_size=bin_s, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_SIZE_MS)
    Y_pc = pca_neural(Y_sm, k=K_PCS)[0]
    traj = trial_average_pc(Y_pc, meta, n_phase_bins=N_PHASE_BINS)
    return X, Y_pc, meta, traj


def sub_mask(meta, N, rng):
    tr = meta["trial_number"].unique()
    pick = rng.choice(tr, size=min(N, len(tr)), replace=False)
    return meta["trial_number"].isin(pick).to_numpy()


def decode(Xtr, Ytr, Xte, Yte, mte, train_mask=None):
    if train_mask is not None:
        Xtr, Ytr = Xtr[train_mask], Ytr[train_mask]
    Xc, pred = kalman_fit_predict(Xtr, Ytr, Xte, Yte, mte)
    return m2_per_trial(Xc, pred, mte)


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
    output_csv = OUT_CSV.with_name(f"{OUT_CSV.stem}{suffix}.csv")
    print(f"loading {len(sessions_r1) + len(sessions_r2)} sessions ...")
    cache = {
        s: load(s, EXCLUDE_TRIALS.get(s, []))
        for s in sessions_r1 + sessions_r2
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for r1, r2 in product(sessions_r1, sessions_r2):
        X1, Y1, m1, t1 = cache[r1]
        X2, Y2, m2, t2 = cache[r2]
        Wa, Wb, ma, mb = cca_align(t1, t2)
        Y1c = apply_alignment(Y1, Wa, ma)   # R1 in canonical
        Y2c = apply_alignment(Y2, Wb, mb)   # R2 in canonical
        n1 = m1["trial_number"].nunique(); n2 = m2["trial_number"].nunique()
        N = min(n1, n2)
        # unmatched (full training)
        f_fwd = decode(X1, Y1c, X2, Y2c, m2)              # R1->R2
        f_rev = decode(X2, Y2c, X1, Y1c, m1)              # R2->R1
        # matched: train both on N trials (subsample), average reps
        mfwd, mrev = [], []
        for _ in range(args.reps):
            mfwd.append(decode(X1, Y1c, X2, Y2c, m2, sub_mask(m1, N, rng)))
            mrev.append(decode(X2, Y2c, X1, Y1c, m1, sub_mask(m2, N, rng)))
        rows.append({"animal": args.animal, "r1": r1[4:12], "r2": r2[4:12],
                     "n1": n1, "n2": n2, "N_matched": N,
                     "unmatched_R1R2": f_fwd, "unmatched_R2R1": f_rev,
                     "matched_R1R2": float(np.mean(mfwd)), "matched_R2R1": float(np.mean(mrev))})
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    um = df["unmatched_R2R1"].mean() - df["unmatched_R1R2"].mean()
    mm = df["matched_R2R1"].mean() - df["matched_R1R2"].mean()
    print(f"\nn pairs = {len(df)};  R1 trials ~{df.n1.mean():.0f}, R2 trials ~{df.n2.mean():.0f}, "
          f"matched N ~{df.N_matched.mean():.0f}")
    print("\n              R1->R2   R2->R1   asymmetry")
    print(f"  unmatched   {df.unmatched_R1R2.mean():.3f}    {df.unmatched_R2R1.mean():.3f}    {um:+.3f}")
    print(f"  matched     {df.matched_R1R2.mean():.3f}    {df.matched_R2R1.mean():.3f}    {mm:+.3f}")
    print(f"\n  asymmetry retained after matching trial count: {mm/um*100:.0f}%")
    print("  -> if ~100%: trial count NOT the cause; if <<100%: trial count drove it")
    print(f"\nsaved {output_csv}")


if __name__ == "__main__":
    main()
