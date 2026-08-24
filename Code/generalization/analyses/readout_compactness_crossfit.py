"""Cross-validated test of whether R2 has a more compact task read-out."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, str(THIS_DIR.parents[1]))

from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2
from nested_cca import fit_pca_projector
from private_readout_crossfit import N_FOLDS, SEED, TARGET, load_session, trial_folds
from readout_compactness import (
    cumulative_predictive_energy,
    fit_predictive_readout,
    predictive_effective_rank,
)

K = 12
N_CALIBRATION_MATCHED = 32
N_EVALUATION_MATCHED = 8
WHITENING_RIDGE = 1e-6
MODES = ("all_available", "trial_matched")

REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "readout_compactness"


def select_trials(meta, allowed, n_trials, rng):
    available = np.asarray(sorted(meta.loc[allowed, "trial_number"].unique()))
    if len(available) < n_trials:
        raise ValueError(f"Requested {n_trials} trials from only {len(available)}")
    selected = rng.choice(available, size=n_trials, replace=False)
    return allowed & meta["trial_number"].isin(selected).to_numpy()


def trial_corr(movement, prediction, meta):
    correlations = []
    for _, indices in meta.groupby("trial_number").indices.items():
        indices = np.asarray(indices)
        if len(indices) < 4:
            continue
        per_dimension = []
        for dim in range(movement.shape[1]):
            value = np.corrcoef(
                movement[indices, dim], prediction[indices, dim]
            )[0, 1]
            per_dimension.append(value)
        correlations.append(np.nanmean(per_dimension))
    return float(np.nanmean(correlations)) if correlations else np.nan


def calibration_scaled_r2(calibration_movement, evaluation_movement, prediction):
    center = calibration_movement.mean(axis=0)
    scale = calibration_movement.var(axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    residual = np.mean(np.square(evaluation_movement - prediction) / scale)
    baseline = np.mean(np.square(evaluation_movement - center) / scale)
    return float(1.0 - residual / baseline) if baseline > 0 else np.nan


def evaluate_split(data, calibration, evaluation):
    projector = fit_pca_projector(data["Y"][calibration], K)
    neural = projector.transform(data["Y"])
    model = fit_predictive_readout(
        neural[calibration], data["X"][calibration], WHITENING_RIDGE
    )
    singular_values = model.singular_values
    evaluation_meta = data["meta"][evaluation].reset_index(drop=True)
    row = {
        "effective_rank": predictive_effective_rank(singular_values),
        "rank1_energy": cumulative_predictive_energy(singular_values, 1),
        "rank2_energy": cumulative_predictive_energy(singular_values, 2),
    }
    for index, value in enumerate(singular_values, start=1):
        row[f"singular_{index}"] = float(value)
    for rank in range(1, data["X"].shape[1] + 1):
        prediction = model.predict(neural[evaluation], rank)
        row[f"corr_rank{rank}"] = trial_corr(
            data["X"][evaluation], prediction, evaluation_meta
        )
        row[f"cv_r2_rank{rank}"] = calibration_scaled_r2(
            data["X"][calibration], data["X"][evaluation], prediction
        )
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.job_index < args.num_jobs:
        raise ValueError("job-index must be in [0, num-jobs)")
    sessions = [
        ("R1", index, session) for index, session in enumerate(SESSIONS_R1)
    ] + [
        ("R2", index, session) for index, session in enumerate(SESSIONS_R2)
    ]
    selected = [
        item for index, item in enumerate(sessions)
        if index % args.num_jobs == args.job_index
    ]
    print(
        f"readout compactness job {args.job_index + 1}/{args.num_jobs}: "
        f"{len(selected)} sessions",
        flush=True,
    )
    rows = []
    for session_position, (epoch, epoch_index, session) in enumerate(selected):
        data = load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        global_index = SESSIONS_R1.index(session) if epoch == "R1" else len(SESSIONS_R1) + SESSIONS_R2.index(session)
        for repeat in range(args.repeats):
            split_seed = SEED + global_index * 100_000 + repeat * 1000
            folds = trial_folds(data["meta"], N_FOLDS, split_seed)
            for fold, evaluation_all in enumerate(folds):
                calibration_all = ~evaluation_all
                for mode_index, mode in enumerate(MODES):
                    calibration = calibration_all
                    evaluation = evaluation_all
                    if mode == "trial_matched":
                        rng = np.random.default_rng(
                            split_seed + fold * 10 + mode_index
                        )
                        calibration = select_trials(
                            data["meta"], calibration_all,
                            N_CALIBRATION_MATCHED, rng
                        )
                        evaluation = select_trials(
                            data["meta"], evaluation_all,
                            N_EVALUATION_MATCHED, rng
                        )
                    row = evaluate_split(data, calibration, evaluation)
                    row.update({
                        "target": TARGET,
                        "epoch": epoch,
                        "epoch_index": epoch_index,
                        "session": session,
                        "mode": mode,
                        "repeat": repeat,
                        "fold": fold,
                        "n_calibration_trials": int(
                            data["meta"].loc[calibration, "trial_number"].nunique()
                        ),
                        "n_evaluation_trials": int(
                            data["meta"].loc[evaluation, "trial_number"].nunique()
                        ),
                        "n_calibration_bins": int(calibration.sum()),
                        "n_evaluation_bins": int(evaluation.sum()),
                    })
                    rows.append(row)
        print(
            f"job {args.job_index + 1}: completed "
            f"{session_position + 1}/{len(selected)} sessions",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / (
        f"readout_compactness_job_{args.job_index}_of_{args.num_jobs}.csv"
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"saved {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
