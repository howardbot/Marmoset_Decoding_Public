"""Repeated target-held-out test of private read-out directions.

Each repeat creates five trial folds in both sessions. Four folds jointly fit
the neural-only PCA/CCA alignment and the kinematic read-out spaces; the fifth
fold is never used until evaluation. The test compares shared-potent projection
and R1/R2-private ablations with rank-matched random spaces drawn inside the
union of the two potent spaces.

The script is shardable by R2 day so all three days can run concurrently.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from nested_cca import NestedCCAAlignment, fit_pca_projector, fit_phase_matched_cca
from nested_cca_validation import load_session
from readout_subspaces import (
    orthogonal_complement,
    principal_readout_subspaces,
    random_subspace_within,
    readout_basis,
)
from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2

K = 12
N_PHASE_BINS = 30
N_FOLDS = 5
L2 = 1e-2
SEED = 20260713
TARGET = "relative_position"
THRESHOLDS = (0.3, 0.5, 0.7)

REPO = _THIS.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "private_readout_crossfit"


def trial_folds(meta: pd.DataFrame, n_folds: int, seed: int) -> list[np.ndarray]:
    trials = np.asarray(sorted(meta["trial_number"].unique()))
    if len(trials) < n_folds * 2:
        raise ValueError("Too few trials for cross-fitting")
    groups = np.array_split(np.random.default_rng(seed).permutation(trials), n_folds)
    return [meta["trial_number"].isin(group).to_numpy() for group in groups]


def fit_calibration_alignment(a, b, calibration_a, calibration_b, seed: int):
    pca_a = fit_pca_projector(a["Y"][calibration_a], K)
    pca_b = fit_pca_projector(b["Y"][calibration_b], K)
    pc_a = pca_a.transform(a["Y"][calibration_a])
    pc_b = pca_b.transform(b["Y"][calibration_b])
    meta_a = a["meta"][calibration_a].reset_index(drop=True)
    meta_b = b["meta"][calibration_b].reset_index(drop=True)
    rotations = fit_phase_matched_cca(
        pc_a,
        meta_a,
        pc_b,
        meta_b,
        n_components=K,
        n_phase_bins=N_PHASE_BINS,
        rng=np.random.default_rng(seed),
    )
    return NestedCCAAlignment(
        train_pca=pca_a,
        target_pca=pca_b,
        train_rotation=rotations[0],
        target_rotation=rotations[1],
        train_cca_mean=rotations[2],
        target_cca_mean=rotations[3],
    )


def standardize_from_calibration(activity: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    mean = activity[calibration].mean(axis=0)
    scale = activity[calibration].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (activity - mean) / scale


def ridge_fit(activity: np.ndarray, movement: np.ndarray) -> np.ndarray:
    return np.linalg.solve(
        activity.T @ activity + L2 * np.eye(activity.shape[1]),
        activity.T @ movement,
    )


def trial_corr(movement: np.ndarray, prediction: np.ndarray, meta: pd.DataFrame) -> float:
    correlations = []
    for _, indices in meta.groupby("trial_number").indices.items():
        indices = np.asarray(indices)
        if len(indices) < 4:
            continue
        per_dimension = [
            np.corrcoef(movement[indices, dim], prediction[indices, dim])[0, 1]
            for dim in range(movement.shape[1])
        ]
        correlations.append(np.nanmean(per_dimension))
    return float(np.nanmean(correlations)) if correlations else np.nan


def score_fixed_map(activity, movement, evaluation, meta, weights) -> float:
    evaluation_meta = meta[evaluation].reset_index(drop=True)
    return trial_corr(
        movement[evaluation], activity[evaluation] @ weights, evaluation_meta
    )


def cross_decode(
    source_activity,
    source_movement,
    source_calibration,
    target_activity,
    target_movement,
    target_evaluation,
    target_meta,
    basis=None,
) -> float:
    if basis is not None:
        source_activity = source_activity @ basis
        target_activity = target_activity @ basis
    weights = ridge_fit(
        source_activity[source_calibration], source_movement[source_calibration]
    )
    evaluation_meta = target_meta[target_evaluation].reset_index(drop=True)
    return trial_corr(
        target_movement[target_evaluation],
        target_activity[target_evaluation] @ weights,
        evaluation_meta,
    )


def directional_scores(a, b, za, zb, calibration_a, calibration_b,
                       evaluation_a, evaluation_b, basis=None) -> tuple[float, float]:
    forward = cross_decode(
        za, a["X"], calibration_a, zb, b["X"], evaluation_b, b["meta"], basis
    )
    reverse = cross_decode(
        zb, b["X"], calibration_b, za, a["X"], evaluation_a, a["meta"], basis
    )
    return forward, reverse


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


def evaluate_split(a, b, calibration_a, calibration_b, evaluation_a, evaluation_b,
                   fit_seed: int, random_draws: int):
    model = fit_calibration_alignment(a, b, calibration_a, calibration_b, fit_seed)
    za = standardize_from_calibration(model.transform_train(a["Y"]), calibration_a)
    zb = standardize_from_calibration(model.transform_target(b["Y"]), calibration_b)

    weights_a = ridge_fit(za[calibration_a], a["X"][calibration_a])
    weights_b = ridge_fit(zb[calibration_b], b["X"][calibration_b])
    basis_a = readout_basis(weights_a)
    basis_b = readout_basis(weights_b)
    forward_full, reverse_full = directional_scores(
        a, b, za, zb, calibration_a, calibration_b, evaluation_a, evaluation_b
    )
    own_a = score_fixed_map(za, a["X"], evaluation_a, a["meta"], weights_a)
    own_b = score_fixed_map(zb, b["X"], evaluation_b, b["meta"], weights_b)
    map_rows = {
        "fwd_cross_map": forward_full,
        "rev_cross_map": reverse_full,
        "own_r1_map": own_a,
        "own_r2_map": own_b,
        "fwd_map_loss": own_b - forward_full,
        "rev_map_loss": own_a - reverse_full,
        "map_loss_asymmetry": (own_b - forward_full) - (own_a - reverse_full),
        "weight_cosine": float(
            np.dot(weights_a.ravel(), weights_b.ravel())
            / (np.linalg.norm(weights_a) * np.linalg.norm(weights_b) + 1e-12)
        ),
    }

    subspace_rows = []
    for threshold_index, threshold in enumerate(THRESHOLDS):
        spaces = principal_readout_subspaces(basis_a, basis_b, threshold)
        shared_rank = spaces.shared.shape[1]
        private_a_rank = spaces.private_a.shape[1]
        private_b_rank = spaces.private_b.shape[1]
        if shared_rank:
            forward_shared, reverse_shared = directional_scores(
                a, b, za, zb, calibration_a, calibration_b,
                evaluation_a, evaluation_b, spaces.shared
            )
        else:
            forward_shared = reverse_shared = np.nan

        keep_without_a = orthogonal_complement(spaces.private_a)
        keep_without_b = orthogonal_complement(spaces.private_b)
        forward_minus_a, reverse_minus_a = directional_scores(
            a, b, za, zb, calibration_a, calibration_b,
            evaluation_a, evaluation_b, keep_without_a
        )
        forward_minus_b, reverse_minus_b = directional_scores(
            a, b, za, zb, calibration_a, calibration_b,
            evaluation_a, evaluation_b, keep_without_b
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
                random_shared = random_subspace_within(
                    spaces.union, shared_rank, rng
                )
                fwd, rev = directional_scores(
                    a, b, za, zb, calibration_a, calibration_b,
                    evaluation_a, evaluation_b, random_shared
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
                evaluation_a, evaluation_b, random_remove_a
            )
            random_values["fwd_random_ablate_a"].append(fwd)
            random_values["rev_random_ablate_a"].append(rev)
            fwd, rev = directional_scores(
                a, b, za, zb, calibration_a, calibration_b,
                evaluation_a, evaluation_b, random_remove_b
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
        subspace_rows.append(row)
    return subspace_rows, map_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-index", type=int, required=True, choices=range(len(SESSIONS_R2)))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-draws", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r2_session = SESSIONS_R2[args.r2_index]
    sessions = list(SESSIONS_R1) + [r2_session]
    print(f"loading shard R2[{args.r2_index}]={r2_session}", flush=True)
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    subspace_rows = []
    map_rows = []
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
                rows, map_row = evaluate_split(
                    a, b, calibration_a, calibration_b, evaluation_a, evaluation_b,
                    split_seed + fold, args.random_draws
                )
                identifiers = {
                    "target": TARGET,
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "repeat": repeat,
                    "fold": fold,
                    "n_calibration_r1": int(a["meta"][calibration_a]["trial_number"].nunique()),
                    "n_calibration_r2": int(b["meta"][calibration_b]["trial_number"].nunique()),
                    "n_evaluation_r1": int(a["meta"][evaluation_a]["trial_number"].nunique()),
                    "n_evaluation_r2": int(b["meta"][evaluation_b]["trial_number"].nunique()),
                }
                for row in rows:
                    row.update(identifiers)
                    subspace_rows.append(row)
                map_row.update(identifiers)
                map_rows.append(map_row)
        print(
            f"shard {args.r2_index}: completed {pair_index + 1}/{len(SESSIONS_R1)} R1 pairs",
            flush=True,
        )

    subspace = pd.DataFrame(subspace_rows)
    maps = pd.DataFrame(map_rows)
    subspace_path = OUT_DIR / f"subspace_shard_{args.r2_index}.csv"
    map_path = OUT_DIR / f"maps_shard_{args.r2_index}.csv"
    subspace.to_csv(subspace_path, index=False)
    maps.to_csv(map_path, index=False)
    print(f"saved {subspace_path} ({len(subspace)} rows)", flush=True)
    print(f"saved {map_path} ({len(maps)} rows)", flush=True)


if __name__ == "__main__":
    main()
