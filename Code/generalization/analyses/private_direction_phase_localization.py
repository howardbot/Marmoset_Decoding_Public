"""Cross-fitted what/when/error localization of R1-private read-out directions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from big_sweep_phase2_crossday import (
    EXCLUDE_TRIALS,
    SESSIONS_R1,
    SESSIONS_R2,
    kalman_fit_predict,
)
from private_direction_localization import (
    FEATURE_NAMES,
    directional_target_scale,
    kinematic_features,
    mean_trial_correlations,
    phase_correlations,
    phase_scaled_mse,
    phase_stack,
    safe_correlation,
)
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
from readout_subspaces import (
    orthogonal_complement,
    principal_readout_subspaces,
    random_subspace_within,
    readout_basis,
)

REPO = _THIS.parents[2]
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "private_direction_localization"
THRESHOLD = 0.5
N_PHASE = 30
BIN_SECONDS = 0.03
DEFAULT_RANDOM_DRAWS = 3


def decode_features(activity, features, calibration, basis):
    if basis.shape[1] == 0:
        return np.full_like(features, np.nan, dtype=float)
    projected = activity @ basis
    center = features[calibration].mean(axis=0)
    scale = features[calibration].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (features - center) / scale
    weights = ridge_fit(projected[calibration], standardized[calibration])
    return projected @ weights * scale + center


def encoding_metrics(truth, prediction, meta):
    overall = mean_trial_correlations(truth, prediction, meta)
    truth_phase, trials = phase_stack(truth, meta, N_PHASE)
    prediction_phase, prediction_trials = phase_stack(prediction, meta, N_PHASE)
    if not np.array_equal(trials, prediction_trials):
        raise AssertionError("truth/prediction phase trials disagree")
    return overall, phase_correlations(truth_phase, prediction_phase)


def kalman_predictions(
    train,
    train_activity,
    calibration,
    target,
    target_activity,
    evaluation,
    basis,
):
    if basis is not None:
        train_activity = train_activity @ basis
        target_activity = target_activity @ basis
    evaluation_meta = target["meta"][evaluation].reset_index(drop=True)
    return kalman_fit_predict(
        train["X"][calibration],
        train_activity[calibration],
        target["X"][evaluation],
        target_activity[evaluation],
        evaluation_meta,
    )


def error_metrics(truth_position, prediction_position, meta, feature_scale):
    truth = kinematic_features(truth_position, meta, BIN_SECONDS)
    prediction = kinematic_features(prediction_position, meta, BIN_SECONDS)
    overall_corr = mean_trial_correlations(truth, prediction, meta)
    overall_nmse = np.mean(((truth - prediction) / feature_scale) ** 2, axis=0)
    truth_phase, trials = phase_stack(truth, meta, N_PHASE)
    prediction_phase, prediction_trials = phase_stack(prediction, meta, N_PHASE)
    if not np.array_equal(trials, prediction_trials):
        raise AssertionError("truth/prediction phase trials disagree")
    return {
        "features": truth,
        "overall_corr": overall_corr,
        "overall_nmse": overall_nmse,
        "phase_corr": phase_correlations(truth_phase, prediction_phase),
        "phase_nmse": phase_scaled_mse(
            truth_phase, prediction_phase, feature_scale
        ),
    }


def append_encoding_rows(
    stores,
    identifiers,
    ranks,
    truth,
    meta,
    predictions,
):
    metrics = {}
    for name, prediction in predictions.items():
        if isinstance(prediction, list):
            draws = [encoding_metrics(truth, draw, meta) for draw in prediction]
            metrics[name] = (
                np.nanmean([draw[0] for draw in draws], axis=0),
                np.nanmean([draw[1] for draw in draws], axis=0),
            )
        else:
            metrics[name] = encoding_metrics(truth, prediction, meta)
    for feature_index, feature in enumerate(FEATURE_NAMES):
        row = {
            **identifiers,
            **ranks,
            "feature": feature,
        }
        for name, (overall, _) in metrics.items():
            row[f"{name}_corr"] = overall[feature_index]
        row["private_excess_random"] = (
            row["private_corr"] - row["random_private_corr"]
        )
        row["shared_excess_random"] = (
            row["shared_corr"] - row["random_shared_corr"]
        )
        stores["encoding_overall"].append(row)
        for phase in range(N_PHASE):
            phase_row = {
                **identifiers,
                **ranks,
                "feature": feature,
                "phase_bin": phase,
                "phase_fraction": phase / (N_PHASE - 1),
            }
            for name, (_, phase_values) in metrics.items():
                phase_row[f"{name}_corr"] = phase_values[phase, feature_index]
            phase_row["private_excess_random"] = (
                phase_row["private_corr"]
                - phase_row["random_private_corr"]
            )
            phase_row["shared_excess_random"] = (
                phase_row["shared_corr"]
                - phase_row["random_shared_corr"]
            )
            stores["encoding_phase"].append(phase_row)


def trial_rows(
    identifiers,
    direction,
    target,
    evaluation,
    target_activity,
    private_basis,
    random_private_bases,
    truth_position,
    full_prediction,
    ablated_prediction,
    feature_scale,
):
    meta = target["meta"][evaluation].reset_index(drop=True)
    raw_position = target["X"][evaluation]
    evaluation_activity = target_activity[evaluation]
    private_projection = evaluation_activity @ private_basis
    random_private_projections = [
        evaluation_activity @ basis for basis in random_private_bases
    ]
    truth_features = kinematic_features(truth_position, meta, BIN_SECONDS)
    full_features = kinematic_features(full_prediction, meta, BIN_SECONDS)
    ablated_features = kinematic_features(ablated_prediction, meta, BIN_SECONDS)
    rows = []
    for trial, indices in meta.groupby("trial_number", sort=False).indices.items():
        indices = np.asarray(indices, dtype=int)
        position_scale = feature_scale[:3]
        full_nmse = float(np.mean(
            ((truth_position[indices] - full_prediction[indices]) / position_scale) ** 2
        ))
        ablated_nmse = float(np.mean(
            ((truth_position[indices] - ablated_prediction[indices]) / position_scale) ** 2
        ))
        full_corr = np.nanmean([
            safe_correlation(
                truth_position[indices, axis], full_prediction[indices, axis]
            )
            for axis in range(3)
        ])
        ablated_corr = np.nanmean([
            safe_correlation(
                truth_position[indices, axis], ablated_prediction[indices, axis]
            )
            for axis in range(3)
        ])
        endpoint = raw_position[indices[-1]]
        rows.append({
            **identifiers,
            "direction": direction,
            "target_trial": trial,
            "result": meta.iloc[indices[0]]["result"],
            "success": float(meta.iloc[indices[0]]["result"] == "S"),
            "duration_s": len(indices) * BIN_SECONDS,
            "mean_speed": float(np.mean(truth_features[indices, 6])),
            "peak_speed": float(np.max(truth_features[indices, 6])),
            "endpoint_x": endpoint[0],
            "endpoint_y": endpoint[1],
            "endpoint_z": endpoint[2],
            "endpoint_distance": float(np.linalg.norm(endpoint)),
            "full_activity_energy": float(np.mean(evaluation_activity[indices] ** 2)),
            "private_energy": float(np.mean(private_projection[indices] ** 2)),
            "random_private_energy": float(np.mean([
                np.mean(projection[indices] ** 2)
                for projection in random_private_projections
            ])),
            "full_position_nmse": full_nmse,
            "ablated_position_nmse": ablated_nmse,
            "position_nmse_rescue": full_nmse - ablated_nmse,
            "full_position_corr": full_corr,
            "ablated_position_corr": ablated_corr,
            "position_corr_rescue": ablated_corr - full_corr,
            "full_speed_nmse": float(np.mean(
                ((truth_features[indices, 6] - full_features[indices, 6])
                 / feature_scale[6]) ** 2
            )),
            "ablated_speed_nmse": float(np.mean(
                ((truth_features[indices, 6] - ablated_features[indices, 6])
                 / feature_scale[6]) ** 2
            )),
        })
    return rows


def append_error_rows(
    stores,
    identifiers,
    direction,
    target,
    evaluation,
    target_activity,
    private_basis,
    random_private_bases,
    feature_scale,
    truth_position,
    full_prediction,
    ablated_prediction,
):
    meta = target["meta"][evaluation].reset_index(drop=True)
    full = error_metrics(
        truth_position, full_prediction, meta, feature_scale
    )
    ablated = error_metrics(
        truth_position, ablated_prediction, meta, feature_scale
    )
    for feature_index, feature in enumerate(FEATURE_NAMES):
        stores["error_overall"].append({
            **identifiers,
            "direction": direction,
            "feature": feature,
            "full_corr": full["overall_corr"][feature_index],
            "ablated_corr": ablated["overall_corr"][feature_index],
            "corr_rescue": (
                ablated["overall_corr"][feature_index]
                - full["overall_corr"][feature_index]
            ),
            "full_nmse": full["overall_nmse"][feature_index],
            "ablated_nmse": ablated["overall_nmse"][feature_index],
            "nmse_rescue": (
                full["overall_nmse"][feature_index]
                - ablated["overall_nmse"][feature_index]
            ),
        })
        for phase in range(N_PHASE):
            stores["error_phase"].append({
                **identifiers,
                "direction": direction,
                "feature": feature,
                "phase_bin": phase,
                "phase_fraction": phase / (N_PHASE - 1),
                "full_corr": full["phase_corr"][phase, feature_index],
                "ablated_corr": ablated["phase_corr"][phase, feature_index],
                "corr_rescue": (
                    ablated["phase_corr"][phase, feature_index]
                    - full["phase_corr"][phase, feature_index]
                ),
                "full_nmse": full["phase_nmse"][phase, feature_index],
                "ablated_nmse": ablated["phase_nmse"][phase, feature_index],
                "nmse_rescue": (
                    full["phase_nmse"][phase, feature_index]
                    - ablated["phase_nmse"][phase, feature_index]
                ),
            })
    stores["trials"].extend(trial_rows(
        identifiers,
        direction,
        target,
        evaluation,
        target_activity,
        private_basis,
        random_private_bases,
        truth_position,
        full_prediction,
        ablated_prediction,
        feature_scale,
    ))


def evaluate_split(
    a,
    b,
    calibration_a,
    calibration_b,
    evaluation_a,
    evaluation_b,
    fit_seed,
    random_draws,
    identifiers,
    stores,
):
    alignment = fit_calibration_alignment(
        a, b, calibration_a, calibration_b, fit_seed
    )
    activity_a = standardize_from_calibration(
        alignment.transform_train(a["Y"]), calibration_a
    )
    activity_b = standardize_from_calibration(
        alignment.transform_target(b["Y"]), calibration_b
    )
    features_a = kinematic_features(a["X"], a["meta"], BIN_SECONDS)
    features_b = kinematic_features(b["X"], b["meta"], BIN_SECONDS)
    feature_scale_a = features_a[calibration_a].std(axis=0)
    feature_scale_b = features_b[calibration_b].std(axis=0)
    feature_scale_a = np.where(feature_scale_a > 1e-8, feature_scale_a, 1.0)
    feature_scale_b = np.where(feature_scale_b > 1e-8, feature_scale_b, 1.0)

    basis_a = readout_basis(
        ridge_fit(activity_a[calibration_a], a["X"][calibration_a])
    )
    basis_b = readout_basis(
        ridge_fit(activity_b[calibration_b], b["X"][calibration_b])
    )
    spaces = principal_readout_subspaces(basis_a, basis_b, THRESHOLD)
    ranks = {
        "rank_shared": spaces.shared.shape[1],
        "rank_private_r1": spaces.private_a.shape[1],
        "rank_private_r2": spaces.private_b.shape[1],
    }

    predictions = {
        "full": decode_features(
            activity_a, features_a, calibration_a, np.eye(activity_a.shape[1])
        ),
        "private": decode_features(
            activity_a, features_a, calibration_a, spaces.private_a
        ),
        "shared": decode_features(
            activity_a, features_a, calibration_a, spaces.shared
        ),
    }
    random_private = []
    random_private_bases = []
    random_shared = []
    for draw in range(random_draws):
        rng = np.random.default_rng(fit_seed + draw * 1000 + 701)
        random_private_basis = random_subspace_within(
            spaces.union, spaces.private_a.shape[1], rng
        )
        random_private_bases.append(random_private_basis)
        random_private.append(decode_features(
            activity_a,
            features_a,
            calibration_a,
            random_private_basis,
        ))
        random_shared.append(decode_features(
            activity_a,
            features_a,
            calibration_a,
            random_subspace_within(
                spaces.union, spaces.shared.shape[1], rng
            ),
        ))
    predictions["random_private"] = random_private
    predictions["random_shared"] = random_shared
    evaluation_meta_a = a["meta"][evaluation_a].reset_index(drop=True)
    append_encoding_rows(
        stores,
        identifiers,
        ranks,
        features_a[evaluation_a],
        evaluation_meta_a,
        {
            name: (
                [draw[evaluation_a] for draw in prediction]
                if isinstance(prediction, list)
                else prediction[evaluation_a]
            )
            for name, prediction in predictions.items()
        },
    )

    keep_without_private = orthogonal_complement(spaces.private_a)
    forward_truth, forward_full = kalman_predictions(
        a, activity_a, calibration_a, b, activity_b, evaluation_b, None
    )
    _, forward_ablated = kalman_predictions(
        a,
        activity_a,
        calibration_a,
        b,
        activity_b,
        evaluation_b,
        keep_without_private,
    )
    append_error_rows(
        stores,
        identifiers,
        "forward",
        b,
        evaluation_b,
        activity_b,
        spaces.private_a,
        random_private_bases,
        directional_target_scale("forward", feature_scale_a, feature_scale_b),
        forward_truth,
        forward_full,
        forward_ablated,
    )

    reverse_truth, reverse_full = kalman_predictions(
        b, activity_b, calibration_b, a, activity_a, evaluation_a, None
    )
    _, reverse_ablated = kalman_predictions(
        b,
        activity_b,
        calibration_b,
        a,
        activity_a,
        evaluation_a,
        keep_without_private,
    )
    append_error_rows(
        stores,
        identifiers,
        "reverse",
        a,
        evaluation_a,
        activity_a,
        spaces.private_a,
        random_private_bases,
        directional_target_scale("reverse", feature_scale_a, feature_scale_b),
        reverse_truth,
        reverse_full,
        reverse_ablated,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-draws", type=int, default=DEFAULT_RANDOM_DRAWS)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.job_index < args.num_jobs:
        raise ValueError("job-index must be in [0, num-jobs)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = [
        (r2_index, pair_index, r1_session, r2_session)
        for r2_index, r2_session in enumerate(SESSIONS_R2)
        for pair_index, r1_session in enumerate(SESSIONS_R1)
    ]
    pairs = [
        pair for index, pair in enumerate(all_pairs)
        if index % args.num_jobs == args.job_index
    ]
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    sessions = sorted({
        session
        for _, _, r1_session, r2_session in pairs
        for session in (r1_session, r2_session)
    })
    print(
        f"loading private localization job {args.job_index + 1}/{args.num_jobs}: "
        f"{len(pairs)} pairs",
        flush=True,
    )
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    stores = {
        "encoding_overall": [],
        "encoding_phase": [],
        "error_overall": [],
        "error_phase": [],
        "trials": [],
    }
    for completed, (r2_index, pair_index, r1_session, r2_session) in enumerate(
        pairs, start=1
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
                identifiers = {
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "repeat": repeat,
                    "fold": fold,
                }
                evaluate_split(
                    a,
                    b,
                    ~evaluation_a,
                    ~evaluation_b,
                    evaluation_a,
                    evaluation_b,
                    split_seed + fold,
                    args.random_draws,
                    identifiers,
                    stores,
                )
        print(
            f"private localization job {args.job_index + 1}/{args.num_jobs}: "
            f"completed {completed}/{len(pairs)}",
            flush=True,
        )
    suffix = f"job_{args.job_index}_of_{args.num_jobs}"
    if args.max_pairs is not None:
        suffix += "_smoke"
    for name, rows in stores.items():
        output = OUT_DIR / f"{name}_{suffix}.csv"
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
