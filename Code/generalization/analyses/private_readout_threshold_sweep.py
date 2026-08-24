"""Cosine-threshold sweep using the original concatenated-transition Kalman."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

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
from private_readout_kalman_decomposition import kalman_score
from readout_subspaces import principal_readout_subspaces, readout_basis

THRESHOLDS = tuple(np.round(np.arange(0.0, 1.0, 0.1), 2))
REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "private_readout_threshold_sweep"


def directional_scores(
    a,
    b,
    za,
    zb,
    calibration_a,
    calibration_b,
    evaluation_a,
    evaluation_b,
    basis,
):
    """Score both directions with the original concatenated-transition Kalman."""
    projected_a = za @ basis
    projected_b = zb @ basis
    return (
        kalman_score(
            a, projected_a, calibration_a, b, projected_b, evaluation_b
        ),
        kalman_score(
            b, projected_b, calibration_b, a, projected_a, evaluation_a
        ),
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
    forward_full = kalman_score(
        a, za, calibration_a, b, zb, evaluation_b
    )
    reverse_full = kalman_score(
        b, zb, calibration_b, a, za, evaluation_a
    )
    full_gap = reverse_full - forward_full

    score_cache = {}
    rows = []
    for threshold in THRESHOLDS:
        spaces = principal_readout_subspaces(basis_a, basis_b, threshold)
        shared_rank = spaces.shared.shape[1]
        shared_mask = tuple((spaces.cosines > threshold).tolist())
        if shared_rank == 0:
            forward_shared = reverse_shared = np.nan
        else:
            if shared_mask not in score_cache:
                score_cache[shared_mask] = directional_scores(
                    a,
                    b,
                    za,
                    zb,
                    calibration_a,
                    calibration_b,
                    evaluation_a,
                    evaluation_b,
                    spaces.shared,
                )
            forward_shared, reverse_shared = score_cache[shared_mask]
        shared_gap = reverse_shared - forward_shared
        closure = full_gap - shared_gap
        rows.append({
            "cosine_threshold": threshold,
            "rank_shared": shared_rank,
            "rank_private_r1": spaces.private_a.shape[1],
            "rank_private_r2": spaces.private_b.shape[1],
            "mean_principal_cosine": float(np.mean(spaces.cosines)),
            "min_principal_cosine": float(np.min(spaces.cosines)),
            "max_principal_cosine": float(np.max(spaces.cosines)),
            "fwd_full": forward_full,
            "rev_full": reverse_full,
            "gap_full": full_gap,
            "fwd_shared": forward_shared,
            "rev_shared": reverse_shared,
            "gap_shared": shared_gap,
            "gap_closure": closure,
            "fraction_closed": closure / full_gap if abs(full_gap) > 1e-12 else np.nan,
            "shared_available": float(shared_rank > 0),
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
    all_pairs = [
        (r2_index, pair_index, r1_session, r2_session)
        for r2_index, r2_session in enumerate(SESSIONS_R2)
        for pair_index, r1_session in enumerate(SESSIONS_R1)
    ]
    selected_pairs = [
        pair for index, pair in enumerate(all_pairs)
        if index % args.num_jobs == args.job_index
    ]
    sessions = sorted({
        session
        for _, _, r1_session, r2_session in selected_pairs
        for session in (r1_session, r2_session)
    })
    print(
        f"original-Kalman threshold job {args.job_index + 1}/{args.num_jobs}: "
        f"{len(selected_pairs)} pairs, {len(sessions)} sessions",
        flush=True,
    )
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    rows = []
    for completed, (r2_index, pair_index, r1_session, r2_session) in enumerate(
        selected_pairs, start=1
    ):
        a, b = cache[r1_session], cache[r2_session]
        for repeat in range(args.repeats):
            split_seed = (
                SEED + r2_index * 1_000_000 + pair_index * 10_000 + repeat * 100
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
                identifiers = {
                    "target": TARGET,
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "repeat": repeat,
                    "fold": fold,
                }
                for row in split_rows:
                    row.update(identifiers)
                    rows.append(row)
        print(
            f"job {args.job_index + 1}: completed "
            f"{completed}/{len(selected_pairs)} pairs",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"threshold_job_{args.job_index}_of_{args.num_jobs}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
