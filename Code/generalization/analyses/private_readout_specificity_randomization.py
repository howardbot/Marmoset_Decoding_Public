"""Conditional randomization test for R1-private read-out specificity."""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, str(THIS_DIR.parents[1]))

from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2
from private_readout_crossfit import (
    N_FOLDS,
    SEED,
    TARGET,
    fit_calibration_alignment,
    load_session,
    ridge_fit,
    standardize_from_calibration,
    trial_folds,
)
from private_readout_kalman_subspaces import directional_scores
from readout_subspaces import (
    orthogonal_complement,
    random_subspace_within,
    readout_basis,
)

THRESHOLD = 0.5
N_OUTPUT_NULL_DRAWS = 20
REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "private_readout_specificity"


def selective_effect(
    a,
    b,
    za,
    zb,
    calibration_a,
    calibration_b,
    evaluation_a,
    evaluation_b,
    full_scores,
    remove_basis,
):
    keep = orthogonal_complement(remove_basis)
    forward, reverse = directional_scores(
        a,
        b,
        za,
        zb,
        calibration_a,
        calibration_b,
        evaluation_a,
        evaluation_b,
        keep,
        transition_mode="trial_aware",
    )
    return float(
        (forward - full_scores[0]) - (reverse - full_scores[1])
    )


def evaluate_split(
    a,
    b,
    calibration_a,
    calibration_b,
    evaluation_a,
    evaluation_b,
    fit_seed,
):
    alignment = fit_calibration_alignment(
        a, b, calibration_a, calibration_b, fit_seed
    )
    za = standardize_from_calibration(
        alignment.transform_train(a["Y"]), calibration_a
    )
    zb = standardize_from_calibration(
        alignment.transform_target(b["Y"]), calibration_b
    )
    basis_a = readout_basis(
        ridge_fit(za[calibration_a], a["X"][calibration_a])
    )
    basis_b = readout_basis(
        ridge_fit(zb[calibration_b], b["X"][calibration_b])
    )
    full_scores = directional_scores(
        a,
        b,
        za,
        zb,
        calibration_a,
        calibration_b,
        evaluation_a,
        evaluation_b,
        transition_mode="trial_aware",
    )

    left, cosines, _ = np.linalg.svd(
        basis_a.T @ basis_b, full_matrices=True
    )
    principal_a = basis_a @ left
    private_indices = tuple(np.flatnonzero(cosines <= THRESHOLD).tolist())
    private_rank = len(private_indices)
    rows = []
    for candidate_index, indices in enumerate(
        combinations(range(principal_a.shape[1]), private_rank)
    ):
        indices = tuple(indices)
        remove = principal_a[:, indices] if indices else np.empty((len(principal_a), 0))
        rows.append({
            "null_family": "r1_potent_principal",
            "candidate_index": candidate_index,
            "is_observed_private": indices == private_indices,
            "selective_effect": selective_effect(
                a, b, za, zb,
                calibration_a, calibration_b, evaluation_a, evaluation_b,
                full_scores, remove,
            ),
            "private_rank": private_rank,
            "mean_principal_cosine": float(np.mean(cosines)),
        })

    output_null = orthogonal_complement(basis_a)
    for draw in range(N_OUTPUT_NULL_DRAWS):
        rng = np.random.default_rng(fit_seed + 1_000_000 + draw * 1000)
        remove = random_subspace_within(output_null, private_rank, rng)
        rows.append({
            "null_family": "r1_output_null",
            "candidate_index": draw,
            "is_observed_private": False,
            "selective_effect": selective_effect(
                a, b, za, zb,
                calibration_a, calibration_b, evaluation_a, evaluation_b,
                full_scores, remove,
            ),
            "private_rank": private_rank,
            "mean_principal_cosine": float(np.mean(cosines)),
        })
    return rows


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
    pairs = [
        (r2_index, pair_index, r1_session, r2_session)
        for r2_index, r2_session in enumerate(SESSIONS_R2)
        for pair_index, r1_session in enumerate(SESSIONS_R1)
    ]
    selected = [
        pair for index, pair in enumerate(pairs)
        if index % args.num_jobs == args.job_index
    ]
    sessions = sorted({session for pair in selected for session in pair[2:]})
    print(
        f"private specificity job {args.job_index + 1}/{args.num_jobs}: "
        f"{len(selected)} pairs, {len(sessions)} sessions",
        flush=True,
    )
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    rows = []
    for selected_index, (r2_index, pair_index, r1_session, r2_session) in enumerate(selected):
        a, b = cache[r1_session], cache[r2_session]
        for repeat in range(args.repeats):
            split_seed = (
                SEED + r2_index * 1_000_000
                + pair_index * 10_000 + repeat * 100
            )
            folds_a = trial_folds(a["meta"], N_FOLDS, split_seed)
            folds_b = trial_folds(b["meta"], N_FOLDS, split_seed + 1)
            for fold in range(N_FOLDS):
                evaluation_a = folds_a[fold]
                evaluation_b = folds_b[fold]
                split_rows = evaluate_split(
                    a,
                    b,
                    ~evaluation_a,
                    ~evaluation_b,
                    evaluation_a,
                    evaluation_b,
                    split_seed + fold,
                )
                for row in split_rows:
                    row.update({
                        "target": TARGET,
                        "r1_session": r1_session,
                        "r2_session": r2_session,
                        "repeat": repeat,
                        "fold": fold,
                    })
                    rows.append(row)
        print(
            f"job {args.job_index + 1}: completed "
            f"{selected_index + 1}/{len(selected)} pairs",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / (
        f"private_specificity_job_{args.job_index}_of_{args.num_jobs}.csv"
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"saved {path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
