"""Is the R2 training advantage due to broader neural-state COVERAGE?

Trial count is ruled out (89% of the asymmetry survives count-matching). R2 has
higher neural trial-to-trial variability (1.80 vs 1.34) -> broader training
coverage -> better generalization is the leading remaining explanation.

Test: for R2->R1, instead of N random R2 trials, train on the N most CENTRAL R2
trials (lowest deviation from the session mean), shrinking R2's training coverage
toward R1's. Compare:
  - R1->R2                      (forward baseline)
  - R2->R1, random N            (count-matched only)
  - R2->R1, central N           (coverage-reduced)
and report the training-set neural variance in each, to confirm coverage was
actually reduced.
  * asymmetry collapses with central-N  -> broader coverage drove it (mundane)
  * asymmetry survives                  -> representational (R2 intrinsically more
                                           generalizable, even at matched coverage)

Config: locked (bin=30, butter_o2, sigma=50ms, K_PCS=15), averaged CCA, velocity.
Output: printed + Results/workflows/manifold_geometry/r2_coverage_control.csv
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
OUT_CSV = REPO_ROOT / "Results" / "workflows" / "manifold_geometry" / "r2_coverage_control.csv"


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
    return X, Y_pc, meta, trial_average_pc(Y_pc, meta, n_phase_bins=N_PHASE_BINS)


def trial_centrality(Yc, meta):
    """Per-trial deviation from session mean (lower = more central). Returns
    {trial: dist} using each trial's mean canonical vector."""
    gm = Yc.mean(0)
    out = {}
    for t, idx in meta.groupby("trial_number").indices.items():
        out[t] = float(np.linalg.norm(Yc[np.asarray(idx)].mean(0) - gm))
    return out


def mask_for(meta, trials):
    return meta["trial_number"].isin(trials).to_numpy()


def train_var(Yc, mask):
    Z = Yc[mask]
    return float(np.trace(np.cov(Z.T)))


def decode(Xtr, Ytr, mask, Xte, Yte, mte):
    Xc, pred = kalman_fit_predict(Xtr[mask], Ytr[mask], Xte, Yte, mte)
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
        X1, Y1, m1, t1 = cache[r1]; X2, Y2, m2, t2 = cache[r2]
        Wa, Wb, ma, mb = cca_align(t1, t2)
        Y1c = apply_alignment(Y1, Wa, ma); Y2c = apply_alignment(Y2, Wb, mb)
        tr1 = list(m1["trial_number"].unique()); tr2 = list(m2["trial_number"].unique())
        N = min(len(tr1), len(tr2))
        cent2 = trial_centrality(Y2c, m2)
        central2 = sorted(tr2, key=lambda t: cent2[t])[:N]      # most-central N R2 trials
        m_central = mask_for(m2, central2)
        fwd, rev_rand, vr1, vr2rand = [], [], [], []
        for _ in range(args.reps):
            s1 = rng.choice(tr1, N, replace=False); s2 = rng.choice(tr2, N, replace=False)
            fwd.append(decode(X1, Y1c, mask_for(m1, s1), X2, Y2c, m2))
            rev_rand.append(decode(X2, Y2c, mask_for(m2, s2), X1, Y1c, m1))
            vr1.append(train_var(Y1c, mask_for(m1, s1)))
            vr2rand.append(train_var(Y2c, mask_for(m2, s2)))
        rev_cent = decode(X2, Y2c, m_central, X1, Y1c, m1)
        rows.append({"animal": args.animal, "r1": r1[4:12], "r2": r2[4:12],
                     "N": N,
                     "R1R2": float(np.mean(fwd)),
                     "R2R1_random": float(np.mean(rev_rand)),
                     "R2R1_central": rev_cent,
                     "var_R1": float(np.mean(vr1)),
                     "var_R2_random": float(np.mean(vr2rand)),
                     "var_R2_central": train_var(Y2c, m_central)})
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    fwd = df.R1R2.mean(); rr = df.R2R1_random.mean(); rc = df.R2R1_central.mean()
    print(f"\nn pairs={len(df)}, N~{df.N.mean():.0f}")
    print("\n=== training-set neural variance (coverage) ===")
    print(f"  R1 (N random)        = {df.var_R1.mean():.3f}")
    print(f"  R2 (N random)        = {df.var_R2_random.mean():.3f}   (R2 broader)")
    print(f"  R2 (N central)       = {df.var_R2_central.mean():.3f}   (shrunk toward R1)")
    print("\n=== cross-day decoding ===")
    print(f"  R1->R2                       = {fwd:.3f}")
    print(f"  R2->R1 (count-matched)       = {rr:.3f}   asym={rr-fwd:+.3f}")
    print(f"  R2->R1 (coverage-reduced)    = {rc:.3f}   asym={rc-fwd:+.3f}")
    if (rr - fwd) != 0:
        print(f"\n  asymmetry retained after coverage reduction: {(rc-fwd)/(rr-fwd)*100:.0f}%")
    print(f"\nsaved {output_csv}")


if __name__ == "__main__":
    main()
