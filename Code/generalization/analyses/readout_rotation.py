"""Did the neural->movement READ-OUT itself rotate across the interference boundary?
(direct test of cause #3, read-out remapping)

The linear decoder W maps canonical neural activity -> kinematics; its row space
is the "read-out subspace" (the neural directions the decoder reads). For each
session pair, in their shared CCA-aligned space (single-trial, K_PCS=15), fit a
full-K read-out on each side and measure the PRINCIPAL ANGLES between the two
read-out subspaces (mean angle in degrees = how much the mapping rotated).

Compare:
  - cross-epoch  R1 vs R2   (39 pairs)
  - within-R1    R1 vs R1'  (sampled baseline)
  - within-R2    0828 vs 0829 (1 pair)
If the cross-epoch rotation exceeds the within-R1 baseline, the read-out rotated
specifically across the interference boundary -> direct evidence the mapping (not
coverage / breadth) changed.

Output: printed summary + CSV. NOTE: n(R2)=3 (within-R2 = 3 pairs).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import mannwhitneyu

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from manifold_align import pca_neural
from dimension_sweep import align_full
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS, EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")
# Also keep 30 ms
BIN_MS = 30
# Smoother parameters
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
# PCA dim, random seed,
K, SEED= 15, 0
TARGETS = ["relative_velocity", "relative_position"]
OUT = _THIS.parents[2] / "Results" / "manifold_geometry"


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
    return X, pca_neural(du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS), k=K)[0], meta, None


def readout(Y, X, l2=1e-2):
    """Full-K linear read-out W (kin x K); return its row-space basis (K x kin)."""
    # Adding bias here, don't have to pass the origin, will be removed later
    A = np.hstack([Y, np.ones((len(Y), 1))])
    # Finding closed-form solution of ridge regression
    W = np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ X)[:-1]   # drop bias row -> (K, kin)
    return W

# ca, cb is the cache for two sessions, rng is the random generator
def rotation_deg(ca, cb, rng):
    # CCA alignment
    Ya, Yb = align_full("single_trial", K, ca, cb, rng)
    if Ya is None:
        return np.nan
    # Training the readout
    Wa, Wb = readout(Ya, ca[0]), readout(Yb, cb[0])
    # return the degree
    return float(np.degrees(np.mean(subspace_angles(Wa, Wb))))


def run_target(target, rng, sessions_r1, sessions_r2):
    cache = {
        s: load(s, target, EXCLUDE_TRIALS.get(s, []))
        for s in sessions_r1 + sessions_r2
    }
    cross = [
        rotation_deg(cache[a], cache[b], rng)
        for a, b in product(sessions_r1, sessions_r2)
    ]
    r1pairs = [
        (a, c) for a, c in product(sessions_r1, sessions_r1) if a != c
    ]
    # ALL R1xR1 pairs -> stable, rng-independent within-R1 rotation baseline
    # (was a random N_WITHIN-pair subsample).
    within1 = [rotation_deg(cache[a], cache[c], rng) for a, c in r1pairs]
    within2 = [
        rotation_deg(cache[a], cache[b], rng)
        for a, b in product(sessions_r2, sessions_r2) if a != b
    ]
    cross = np.array([x for x in cross if np.isfinite(x)])
    within1 = np.array([x for x in within1 if np.isfinite(x)])
    within2 = np.array([x for x in within2 if np.isfinite(x)])
    p = mannwhitneyu(cross, within1, alternative="greater").pvalue
    return cross, within1, within2, p


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
    output_csv = OUT / f"readout_rotation{suffix}.csv"
    rng = np.random.default_rng(SEED)
    rows = []
    for tgt in TARGETS:
        cross, w1, w2, p = run_target(
            tgt, rng, sessions_r1, sessions_r2
        )
        print(f"\n########## {tgt}  (read-out subspace rotation, degrees) ##########")
        print(f"  cross-epoch R1 vs R2 : mean {cross.mean():5.1f}  median {np.median(cross):5.1f}  (n={len(cross)})")
        print(f"  within-R1   R1 vs R1': mean {w1.mean():5.1f}  median {np.median(w1):5.1f}  (n={len(w1)})")
        if len(w2):
            print(f"  within-R2: {w2.mean():5.1f}  (n={len(w2)})")
        diff = cross.mean() - w1.mean()
        # Verdict on EFFECT SIZE (degrees of excess rotation). The Mann-Whitney p is over
        # day-pairs, which are pseudo-replicated (same 14 R1 / 3 R2 sessions recombine) at
        # n(R2)=3, so it is descriptive (flagged p*), NOT valid inference.
        print(f"  cross - within-R1 = {diff:+.1f} deg   Mann-Whitney (cross>within1) p*={p:.4f}")
        verdict = (f"excess rotation across the boundary (+{diff:.1f} deg)" if diff > 5
                   else "no meaningful excess rotation vs within-R1 baseline")
        print(
            f"  => {verdict}   (* pseudo-replicated day-pairs, "
            f"n(R2)={len(sessions_r2)} — descriptive only)"
        )
        rows += [
            dict(
                animal=args.animal,
                target=tgt,
                comparison="cross_R1R2",
                deg=v,
            )
            for v in cross
        ]
        rows += [
            dict(
                animal=args.animal,
                target=tgt,
                comparison="within_R1",
                deg=v,
            )
            for v in w1
        ]
        rows += [
            dict(
                animal=args.animal,
                target=tgt,
                comparison="within_R2",
                deg=v,
            )
            for v in w2
        ]
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"\nsaved {output_csv}")


if __name__ == "__main__":
    main()
