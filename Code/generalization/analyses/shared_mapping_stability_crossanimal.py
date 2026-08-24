"""Cross-animal test of the R2-stable shared neural-behavior map hypothesis.

Each session is evaluated in its own calibration-only PCA coordinates with
fixed, equal trial counts (32 calibration and 8 evaluation trials).  The test
measures held-out decoding quality and split-half readout reproducibility.
It therefore asks whether R2 is a cleaner source model before any cross-day
decoder is applied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, str(THIS_DIR.parents[1]))
sys.path.insert(0, str(THIS_DIR))

from big_sweep_phase2_crossday import ANIMAL_SESSIONS, EXCLUDE_TRIALS, m2_per_trial
from decoder_model_audit import RIDGE_ALPHA, TARGETS, _fit_linear, _predict_linear
from nested_cca import fit_pca_projector
from private_readout_crossfit import (
    N_FOLDS,
    SEED,
    load_session,
    standardize_from_calibration,
    trial_folds,
)
from readout_compactness_crossfit import calibration_scaled_r2, select_trials


K = 12
N_CALIBRATION_TRIALS = 32
N_EVALUATION_TRIALS = 8
REPEATS = 5
REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "shared_mapping_stability"


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return np.nan
    return float(first @ second / denominator)


def split_calibration_trials(
    meta: pd.DataFrame, calibration: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    trials = np.asarray(sorted(meta.loc[calibration, "trial_number"].unique()))
    if len(trials) < 4:
        raise ValueError("at least four calibration trials are required")
    first_trials, second_trials = np.array_split(
        np.random.default_rng(seed).permutation(trials), 2
    )
    first = calibration & meta["trial_number"].isin(first_trials).to_numpy()
    second = calibration & meta["trial_number"].isin(second_trials).to_numpy()
    return first, second


def evaluate_split(
    data,
    calibration: np.ndarray,
    evaluation: np.ndarray,
    split_seed: int,
) -> dict:
    projector = fit_pca_projector(data["Y"][calibration], K)
    neural = standardize_from_calibration(
        projector.transform(data["Y"]), calibration
    )
    state = np.asarray(data["X"], dtype=float)
    coefficients = _fit_linear(
        neural[calibration], state[calibration], RIDGE_ALPHA
    )
    prediction = _predict_linear(neural[evaluation], coefficients)
    evaluation_meta = data["meta"][evaluation].reset_index(drop=True)
    within_corr = m2_per_trial(
        state[evaluation], prediction, evaluation_meta
    )
    cv_r2 = calibration_scaled_r2(
        state[calibration], state[evaluation], prediction
    )

    first, second = split_calibration_trials(
        data["meta"], calibration, split_seed
    )
    coefficients_first = _fit_linear(
        neural[first], state[first], RIDGE_ALPHA
    )
    coefficients_second = _fit_linear(
        neural[second], state[second], RIDGE_ALPHA
    )
    prediction_first = _predict_linear(
        neural[evaluation], coefficients_first
    )
    prediction_second = _predict_linear(
        neural[evaluation], coefficients_second
    )
    prediction_agreement = m2_per_trial(
        prediction_first, prediction_second, evaluation_meta
    )
    return {
        "within_corr": within_corr,
        "cv_r2": cv_r2,
        "weight_cosine": cosine_similarity(
            coefficients_first[1:], coefficients_second[1:]
        ),
        "prediction_agreement": prediction_agreement,
        "half1_corr": m2_per_trial(
            state[evaluation], prediction_first, evaluation_meta
        ),
        "half2_corr": m2_per_trial(
            state[evaluation], prediction_second, evaluation_meta
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=sorted(ANIMAL_SESSIONS), required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-sessions", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = ANIMAL_SESSIONS[args.animal]
    sessions = [
        ("R1", index, session) for index, session in enumerate(sessions_r1)
    ] + [
        ("R2", index, session) for index, session in enumerate(sessions_r2)
    ]
    sessions = sessions[: args.max_sessions]
    print(
        f"shared mapping animal={args.animal} target={args.target} "
        f"sessions={len(sessions)} repeats={args.repeats}",
        flush=True,
    )
    rows = []
    animal_offset = 0 if args.animal == "TS" else 50_000_000
    for session_index, (epoch, epoch_index, session) in enumerate(sessions):
        data = load_session(
            session, args.target, EXCLUDE_TRIALS.get(session, ())
        )
        for repeat in range(args.repeats):
            split_seed = (
                SEED + animal_offset + session_index * 100_000 + repeat * 1000
            )
            folds = trial_folds(data["meta"], N_FOLDS, split_seed)
            for fold, evaluation_pool in enumerate(folds):
                rng = np.random.default_rng(split_seed + fold * 10)
                calibration = select_trials(
                    data["meta"], ~evaluation_pool,
                    N_CALIBRATION_TRIALS, rng,
                )
                evaluation = select_trials(
                    data["meta"], evaluation_pool,
                    N_EVALUATION_TRIALS, rng,
                )
                row = evaluate_split(
                    data, calibration, evaluation, split_seed + fold
                )
                row.update({
                    "animal": args.animal,
                    "target": args.target,
                    "epoch": epoch,
                    "epoch_index": epoch_index,
                    "session": session,
                    "repeat": repeat,
                    "fold": fold,
                    "n_calibration_trials": N_CALIBRATION_TRIALS,
                    "n_evaluation_trials": N_EVALUATION_TRIALS,
                })
                rows.append(row)
        print(
            f"shared mapping {args.animal} {args.target}: "
            f"{session_index + 1}/{len(sessions)} sessions",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.max_sessions is not None or args.repeats != REPEATS:
        suffix = "_smoke"
    output = OUT_DIR / (
        f"mapping_stability_{args.animal.lower()}_{args.target}{suffix}.csv"
    )
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
