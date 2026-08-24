"""Kalman ablation test for ridge-defined private read-out directions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

import numpy as np
import pandas as pd

from kalman_components import fit_kalman_components, predict_kalman_trials
from private_readout_crossfit import (
    N_FOLDS,
    OUT_DIR,
    SEED,
    TARGET,
    THRESHOLDS,
    fit_calibration_alignment,
    load_session,
    mean_std,
    ridge_fit,
    standardize_from_calibration,
    trial_folds,
)
from private_readout_kalman_decomposition import kalman_score
from readout_subspaces import (
    orthogonal_complement,
    principal_readout_subspaces,
    random_subspace_within,
    readout_basis,
)
from big_sweep_phase2_crossday import (
    EXCLUDE_TRIALS,
    SESSIONS_R1,
    SESSIONS_R2,
    m2_per_trial,
)

TRANSITION_MODES = ("concatenated", "trial_aware")


def trial_aware_kalman_score(
    train,
    train_activity,
    train_mask,
    target,
    target_activity,
    evaluation_mask,
) -> float:
    """Fit A/W within trials and score one transfer direction."""
    center = train["X"][train_mask].mean(axis=0)
    train_state = train["X"] - center
    target_state = target["X"] - center
    calibration_meta = train["meta"][train_mask].reset_index(drop=True)
    model = fit_kalman_components(
        train_activity[train_mask],
        train_state[train_mask],
        calibration_meta,
    )
    evaluation_meta = target["meta"][evaluation_mask].reset_index(drop=True)
    evaluation_state = target_state[evaluation_mask]
    prediction = predict_kalman_trials(
        model,
        target_activity[evaluation_mask],
        evaluation_state,
        evaluation_meta,
    )
    return m2_per_trial(evaluation_state, prediction, evaluation_meta)


def directional_scores(a, b, za, zb, calibration_a, calibration_b,
                       evaluation_a, evaluation_b, basis=None,
                       transition_mode="concatenated"):
    if basis is not None:
        za = za @ basis
        zb = zb @ basis
    score = (
        trial_aware_kalman_score
        if transition_mode == "trial_aware"
        else kalman_score
    )
    return (
        score(a, za, calibration_a, b, zb, evaluation_b),
        score(b, zb, calibration_b, a, za, evaluation_a),
    )


def evaluate_split(a, b, calibration_a, calibration_b, evaluation_a, evaluation_b,
                   fit_seed: int, random_draws: int,
                   transition_mode: str = "concatenated"):
    model = fit_calibration_alignment(a, b, calibration_a, calibration_b, fit_seed)
    za = standardize_from_calibration(model.transform_train(a["Y"]), calibration_a)
    zb = standardize_from_calibration(model.transform_target(b["Y"]), calibration_b)
    basis_a = readout_basis(ridge_fit(za[calibration_a], a["X"][calibration_a]))
    basis_b = readout_basis(ridge_fit(zb[calibration_b], b["X"][calibration_b]))
    forward_full, reverse_full = directional_scores(
        a, b, za, zb, calibration_a, calibration_b, evaluation_a, evaluation_b,
        transition_mode=transition_mode,
    )

    rows = []
    for threshold_index, threshold in enumerate(THRESHOLDS):
        spaces = principal_readout_subspaces(basis_a, basis_b, threshold)
        shared_rank = spaces.shared.shape[1]
        private_a_rank = spaces.private_a.shape[1]
        private_b_rank = spaces.private_b.shape[1]
        if shared_rank:
            forward_shared, reverse_shared = directional_scores(
                a, b, za, zb, calibration_a, calibration_b,
                evaluation_a, evaluation_b, spaces.shared,
                transition_mode=transition_mode,
            )
        else:
            forward_shared = reverse_shared = np.nan
        forward_minus_a, reverse_minus_a = directional_scores(
            a, b, za, zb, calibration_a, calibration_b, evaluation_a, evaluation_b,
            orthogonal_complement(spaces.private_a),
            transition_mode=transition_mode,
        )
        forward_minus_b, reverse_minus_b = directional_scores(
            a, b, za, zb, calibration_a, calibration_b, evaluation_a, evaluation_b,
            orthogonal_complement(spaces.private_b),
            transition_mode=transition_mode,
        )

        random_values = {
            "fwd_random_shared": [], "rev_random_shared": [],
            "fwd_random_ablate_a": [], "rev_random_ablate_a": [],
            "fwd_random_ablate_b": [], "rev_random_ablate_b": [],
        }
        for draw in range(random_draws):
            rng = np.random.default_rng(
                fit_seed + threshold_index * 100_000 + draw * 1000
            )
            if shared_rank:
                random_shared = random_subspace_within(spaces.union, shared_rank, rng)
                fwd, rev = directional_scores(
                    a, b, za, zb, calibration_a, calibration_b,
                    evaluation_a, evaluation_b, random_shared,
                    transition_mode=transition_mode,
                )
                random_values["fwd_random_shared"].append(fwd)
                random_values["rev_random_shared"].append(rev)
            random_remove_a = orthogonal_complement(
                random_subspace_within(spaces.union, private_a_rank, rng)
            )
            random_remove_b = orthogonal_complement(
                random_subspace_within(spaces.union, private_b_rank, rng)
            )
            fwd, rev = directional_scores(
                a, b, za, zb, calibration_a, calibration_b,
                evaluation_a, evaluation_b, random_remove_a,
                transition_mode=transition_mode,
            )
            random_values["fwd_random_ablate_a"].append(fwd)
            random_values["rev_random_ablate_a"].append(rev)
            fwd, rev = directional_scores(
                a, b, za, zb, calibration_a, calibration_b,
                evaluation_a, evaluation_b, random_remove_b,
                transition_mode=transition_mode,
            )
            random_values["fwd_random_ablate_b"].append(fwd)
            random_values["rev_random_ablate_b"].append(rev)

        row = {
            "cosine_threshold": threshold,
            "rank_r1": basis_a.shape[1],
            "rank_r2": basis_b.shape[1],
            "rank_shared": shared_rank,
            "rank_private_r1": private_a_rank,
            "rank_private_r2": private_b_rank,
            "mean_principal_cosine": float(np.mean(spaces.cosines)),
            "transition_mode": transition_mode,
            "fwd_full": forward_full,
            "rev_full": reverse_full,
            "fwd_shared": forward_shared,
            "rev_shared": reverse_shared,
            "fwd_minus_r1_private": forward_minus_a,
            "rev_minus_r1_private": reverse_minus_a,
            "fwd_minus_r2_private": forward_minus_b,
            "rev_minus_r2_private": reverse_minus_b,
        }
        for name, values in random_values.items():
            if values:
                row[f"{name}_mean"], row[f"{name}_std"] = mean_std(values)
            else:
                row[f"{name}_mean"] = row[f"{name}_std"] = np.nan
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--r2-index", type=int, choices=range(len(SESSIONS_R2)))
    group.add_argument("--job-index", type=int)
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-draws", type=int, default=3)
    parser.add_argument(
        "--transition-mode",
        choices=TRANSITION_MODES,
        default="concatenated",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = [
        (r2_index, pair_index, r1_session, r2_session)
        for r2_index, r2_session in enumerate(SESSIONS_R2)
        for pair_index, r1_session in enumerate(SESSIONS_R1)
    ]
    if args.r2_index is not None:
        selected_pairs = [pair for pair in all_pairs if pair[0] == args.r2_index]
        mode_tag = "_trial_aware" if args.transition_mode == "trial_aware" else ""
        output = OUT_DIR / f"kalman_subspace{mode_tag}_shard_{args.r2_index}.csv"
        label = f"{args.transition_mode} R2 shard {args.r2_index}"
    else:
        if not 0 <= args.job_index < args.num_jobs:
            raise ValueError("job-index must be in [0, num-jobs)")
        selected_pairs = [
            pair for index, pair in enumerate(all_pairs)
            if index % args.num_jobs == args.job_index
        ]
        mode_tag = "_trial_aware" if args.transition_mode == "trial_aware" else ""
        output = OUT_DIR / (
            f"kalman_subspace{mode_tag}_job_{args.job_index}_of_{args.num_jobs}.csv"
        )
        label = (
            f"{args.transition_mode} job {args.job_index + 1}/{args.num_jobs}"
        )
    sessions = sorted({
        session
        for _, _, r1_session, r2_session in selected_pairs
        for session in (r1_session, r2_session)
    })
    print(
        f"loading Kalman subspace {label}: {len(selected_pairs)} pairs, "
        f"{len(sessions)} sessions",
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
            split_seed = SEED + r2_index * 1_000_000 + pair_index * 10_000 + repeat * 100
            folds_a = trial_folds(a["meta"], N_FOLDS, split_seed)
            folds_b = trial_folds(b["meta"], N_FOLDS, split_seed + 1)
            for fold in range(N_FOLDS):
                evaluation_a = folds_a[fold]
                evaluation_b = folds_b[fold]
                calibration_a = ~evaluation_a
                calibration_b = ~evaluation_b
                split_rows = evaluate_split(
                    a, b, calibration_a, calibration_b, evaluation_a, evaluation_b,
                    split_seed + fold, args.random_draws, args.transition_mode
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
            f"Kalman subspace {label}: "
            f"completed {completed}/{len(selected_pairs)} pairs",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
