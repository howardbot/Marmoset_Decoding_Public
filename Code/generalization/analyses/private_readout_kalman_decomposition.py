"""Kalman sensitivity for target-ceiling versus transfer-penalty decomposition."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from private_readout_crossfit import (
    N_FOLDS,
    OUT_DIR,
    SEED,
    TARGET,
    fit_calibration_alignment,
    load_session,
    standardize_from_calibration,
    trial_folds,
)
from big_sweep_phase2_crossday import (
    EXCLUDE_TRIALS,
    SESSIONS_R1,
    SESSIONS_R2,
    kalman_fit_predict,
    m2_per_trial,
)


def kalman_score(train, train_activity, train_mask, target, target_activity,
                 evaluation_mask) -> float:
    evaluation_meta = target["meta"][evaluation_mask].reset_index(drop=True)
    movement, prediction = kalman_fit_predict(
        train["X"][train_mask],
        train_activity[train_mask],
        target["X"][evaluation_mask],
        target_activity[evaluation_mask],
        evaluation_meta,
    )
    return m2_per_trial(movement, prediction, evaluation_meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-index", type=int, required=True, choices=range(len(SESSIONS_R2)))
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r2_session = SESSIONS_R2[args.r2_index]
    sessions = list(SESSIONS_R1) + [r2_session]
    print(f"loading Kalman shard R2[{args.r2_index}]={r2_session}", flush=True)
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    rows = []
    for pair_index, r1_session in enumerate(SESSIONS_R1):
        a, b = cache[r1_session], cache[r2_session]
        for repeat in range(args.repeats):
            split_seed = SEED + args.r2_index * 1_000_000 + pair_index * 10_000 + repeat * 100
            folds_a = trial_folds(a["meta"], N_FOLDS, split_seed)
            folds_b = trial_folds(b["meta"], N_FOLDS, split_seed + 1)
            for fold in range(N_FOLDS):
                evaluation_a = folds_a[fold]
                evaluation_b = folds_b[fold]
                calibration_a = ~evaluation_a
                calibration_b = ~evaluation_b
                model = fit_calibration_alignment(
                    a, b, calibration_a, calibration_b, split_seed + fold
                )
                za = standardize_from_calibration(
                    model.transform_train(a["Y"]), calibration_a
                )
                zb = standardize_from_calibration(
                    model.transform_target(b["Y"]), calibration_b
                )
                forward = kalman_score(
                    a, za, calibration_a, b, zb, evaluation_b
                )
                reverse = kalman_score(
                    b, zb, calibration_b, a, za, evaluation_a
                )
                own_a = kalman_score(
                    a, za, calibration_a, a, za, evaluation_a
                )
                own_b = kalman_score(
                    b, zb, calibration_b, b, zb, evaluation_b
                )
                rows.append({
                    "target": TARGET,
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "repeat": repeat,
                    "fold": fold,
                    "fwd_cross_map": forward,
                    "rev_cross_map": reverse,
                    "own_r1_map": own_a,
                    "own_r2_map": own_b,
                    "fwd_map_loss": own_b - forward,
                    "rev_map_loss": own_a - reverse,
                    "map_loss_asymmetry": (own_b - forward) - (own_a - reverse),
                })
        print(
            f"Kalman shard {args.r2_index}: completed {pair_index + 1}/{len(SESSIONS_R1)} R1 pairs",
            flush=True,
        )
    output = OUT_DIR / f"kalman_maps_shard_{args.r2_index}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
