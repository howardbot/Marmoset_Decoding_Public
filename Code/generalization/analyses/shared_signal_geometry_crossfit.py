"""Cross-fitted shared-space signal/noise geometry and spectrum mediation."""
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
from private_readout_kalman_subspaces import trial_aware_kalman_score
from readout_compactness import (
    cumulative_predictive_energy,
    fit_predictive_readout,
    predictive_effective_rank,
)
from readout_compactness_crossfit import (
    calibration_scaled_r2,
    select_trials,
    trial_corr,
)
from readout_subspaces import principal_readout_subspaces, readout_basis
from shared_signal_geometry import (
    covariance_spectrum_metrics,
    encoded_covariances,
    fit_generalized_axes,
    fit_movement_encoding,
    heldout_generalized_metrics,
    phase_geometry,
    phase_stack,
    pooled_spectrum_scaling,
    random_orthogonal,
    scaling_transform,
    transform_encoded_components,
    transform_phase_components,
)

THRESHOLD = 0.5
N_PHASE_BINS = 30
N_CALIBRATION_TRIALS = 32
N_EVALUATION_TRIALS = 8
REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "shared_signal_geometry"
PHASE_MODES = (
    "phase_signal_spectrum",
    "phase_residual_spectrum",
    "phase_both_spectra",
)
MOVEMENT_MODES = (
    "movement_signal_spectrum",
    "movement_residual_spectrum",
    "movement_both_spectra",
)
MODES = PHASE_MODES + MOVEMENT_MODES


def matched_masks(data, evaluation_pool, seed):
    calibration_pool = ~evaluation_pool
    rng = np.random.default_rng(seed)
    calibration = select_trials(
        data["meta"], calibration_pool, N_CALIBRATION_TRIALS, rng
    )
    evaluation = select_trials(
        data["meta"], evaluation_pool, N_EVALUATION_TRIALS, rng
    )
    return calibration, evaluation


def curve_reliability(calibration_curve, evaluation_curve):
    first = np.asarray(calibration_curve, dtype=float).ravel()
    second = np.asarray(evaluation_curve, dtype=float).ravel()
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def geometry_metrics(activity, meta, calibration, evaluation):
    calibration_geometry = phase_geometry(
        phase_stack(activity, meta, calibration, N_PHASE_BINS)
    )
    evaluation_geometry = phase_geometry(
        phase_stack(activity, meta, evaluation, N_PHASE_BINS)
    )
    signal_metrics = covariance_spectrum_metrics(
        evaluation_geometry.signal_covariance
    )
    noise_metrics = covariance_spectrum_metrics(
        evaluation_geometry.noise_covariance
    )
    axes, _ = fit_generalized_axes(
        calibration_geometry.signal_covariance,
        calibration_geometry.noise_covariance,
    )
    generalized = heldout_generalized_metrics(
        evaluation_geometry.signal_covariance,
        evaluation_geometry.noise_covariance,
        axes,
    )
    metrics = {
        "phase_signal_power": signal_metrics["power"],
        "phase_signal_effective_rank": signal_metrics["effective_rank"],
        "phase_signal_top1_fraction": signal_metrics["top1_fraction"],
        "phase_residual_power": noise_metrics["power"],
        "phase_residual_effective_rank": noise_metrics["effective_rank"],
        "phase_reliability": curve_reliability(
            calibration_geometry.mean_curve,
            evaluation_geometry.mean_curve,
        ),
        "phase_trace_snr": float(
            np.trace(evaluation_geometry.signal_covariance)
            / max(np.trace(evaluation_geometry.noise_covariance), 1e-12)
        ),
        **{f"phase_{key}": value for key, value in generalized.items()},
    }
    return metrics, calibration_geometry


def movement_geometry_metrics(
    activity, movement, meta, calibration, evaluation
):
    encoding = fit_movement_encoding(
        activity, movement, meta, calibration
    )
    prediction = encoding.predict(movement, meta)
    calibration_signal, calibration_residual = encoded_covariances(
        activity, prediction, calibration
    )
    evaluation_signal, evaluation_residual = encoded_covariances(
        activity, prediction, evaluation
    )
    signal_metrics = covariance_spectrum_metrics(evaluation_signal)
    residual_metrics = covariance_spectrum_metrics(evaluation_residual)
    axes, _ = fit_generalized_axes(
        calibration_signal, calibration_residual
    )
    generalized = heldout_generalized_metrics(
        evaluation_signal, evaluation_residual, axes
    )
    signal_trace = float(np.trace(evaluation_signal))
    residual_trace = float(np.trace(evaluation_residual))
    metrics = {
        "movement_signal_power": signal_metrics["power"],
        "movement_signal_effective_rank": signal_metrics["effective_rank"],
        "movement_signal_top1_fraction": signal_metrics["top1_fraction"],
        "movement_residual_power": residual_metrics["power"],
        "movement_residual_effective_rank": residual_metrics["effective_rank"],
        "movement_trace_snr": signal_trace / max(residual_trace, 1e-12),
        "movement_encoding_fraction": (
            signal_trace / max(signal_trace + residual_trace, 1e-12)
        ),
        **{f"movement_{key}": value for key, value in generalized.items()},
    }
    return (
        metrics,
        prediction,
        calibration_signal,
        calibration_residual,
    )


def predictive_metrics(activity, movement, meta, calibration, evaluation):
    model = fit_predictive_readout(activity[calibration], movement[calibration])
    singular_values = model.singular_values
    rank = len(singular_values)
    prediction_rank1 = model.predict(activity[evaluation], rank=1)
    prediction_full = model.predict(activity[evaluation], rank=rank)
    evaluation_meta = meta[evaluation].reset_index(drop=True)
    corr_rank1 = trial_corr(
        movement[evaluation], prediction_rank1, evaluation_meta
    )
    corr_full = trial_corr(
        movement[evaluation], prediction_full, evaluation_meta
    )
    return {
        "predictive_effective_rank": predictive_effective_rank(singular_values),
        "predictive_rank1_energy": cumulative_predictive_energy(
            singular_values, 1
        ),
        "predictive_corr_rank1": corr_rank1,
        "predictive_corr_full": corr_full,
        "predictive_rank1_fraction": float(corr_rank1 / corr_full)
        if abs(corr_full) > 1e-12
        else np.nan,
        "predictive_cv_r2_rank1": calibration_scaled_r2(
            movement[calibration], movement[evaluation], prediction_rank1
        ),
        "predictive_cv_r2_full": calibration_scaled_r2(
            movement[calibration], movement[evaluation], prediction_full
        ),
    }


def directional_transformed_scores(
    a,
    b,
    shared_a,
    shared_b,
    calibration_a,
    calibration_b,
    evaluation_a,
    evaluation_b,
    geometry_a,
    geometry_b,
    transforms_ab,
    transforms_ba,
):
    transformed_a = transform_phase_components(
        shared_a,
        a["meta"],
        calibration_a,
        geometry_a.mean_curve,
        transforms_ab[0],
        transforms_ab[1],
    )
    transformed_b = transform_phase_components(
        shared_b,
        b["meta"],
        calibration_b,
        geometry_b.mean_curve,
        transforms_ba[0],
        transforms_ba[1],
    )
    forward = trial_aware_kalman_score(
        a,
        transformed_a,
        calibration_a,
        b,
        shared_b,
        evaluation_b,
    )
    reverse = trial_aware_kalman_score(
        b,
        transformed_b,
        calibration_b,
        a,
        shared_a,
        evaluation_a,
    )
    return forward, reverse


def mode_transforms(signal_transform, noise_transform, mode):
    if mode.endswith("signal_spectrum"):
        return signal_transform, None
    if mode.endswith("residual_spectrum"):
        return None, noise_transform
    if mode.endswith("both_spectra"):
        return signal_transform, noise_transform
    raise ValueError(f"unknown mediation mode: {mode}")


def evaluate_split(
    a,
    b,
    calibration_a,
    calibration_b,
    evaluation_a,
    evaluation_b,
    fit_seed,
    random_draws,
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
    spaces = principal_readout_subspaces(basis_a, basis_b, THRESHOLD)
    rank = spaces.shared.shape[1]
    row = {"shared_rank": rank, "mean_principal_cosine": float(np.mean(spaces.cosines))}
    if rank == 0:
        return row, []

    shared_a = za @ spaces.shared
    shared_b = zb @ spaces.shared
    metrics_a, geometry_a = geometry_metrics(
        shared_a, a["meta"], calibration_a, evaluation_a
    )
    metrics_b, geometry_b = geometry_metrics(
        shared_b, b["meta"], calibration_b, evaluation_b
    )
    movement_metrics_a, movement_prediction_a, movement_signal_a, movement_residual_a = (
        movement_geometry_metrics(
            shared_a, a["X"], a["meta"], calibration_a, evaluation_a
        )
    )
    movement_metrics_b, movement_prediction_b, movement_signal_b, movement_residual_b = (
        movement_geometry_metrics(
            shared_b, b["X"], b["meta"], calibration_b, evaluation_b
        )
    )
    predictive_a = predictive_metrics(
        shared_a, a["X"], a["meta"], calibration_a, evaluation_a
    )
    predictive_b = predictive_metrics(
        shared_b, b["X"], b["meta"], calibration_b, evaluation_b
    )
    row.update({f"r1_{key}": value for key, value in metrics_a.items()})
    row.update({f"r2_{key}": value for key, value in metrics_b.items()})
    row.update({f"r1_{key}": value for key, value in movement_metrics_a.items()})
    row.update({f"r2_{key}": value for key, value in movement_metrics_b.items()})
    row.update({f"r1_{key}": value for key, value in predictive_a.items()})
    row.update({f"r2_{key}": value for key, value in predictive_b.items()})

    forward_shared = trial_aware_kalman_score(
        a, shared_a, calibration_a, b, shared_b, evaluation_b
    )
    reverse_shared = trial_aware_kalman_score(
        b, shared_b, calibration_b, a, shared_a, evaluation_a
    )
    row.update({
        "fwd_shared": forward_shared,
        "rev_shared": reverse_shared,
        "gap_shared": reverse_shared - forward_shared,
    })

    signal_axes_ab, signal_scales_ab = pooled_spectrum_scaling(
        geometry_a.signal_covariance, geometry_b.signal_covariance
    )
    signal_axes_ba, signal_scales_ba = pooled_spectrum_scaling(
        geometry_b.signal_covariance, geometry_a.signal_covariance
    )
    noise_axes_ab, noise_scales_ab = pooled_spectrum_scaling(
        geometry_a.noise_covariance, geometry_b.noise_covariance
    )
    noise_axes_ba, noise_scales_ba = pooled_spectrum_scaling(
        geometry_b.noise_covariance, geometry_a.noise_covariance
    )
    signal_transform_ab = scaling_transform(signal_axes_ab, signal_scales_ab)
    signal_transform_ba = scaling_transform(signal_axes_ba, signal_scales_ba)
    noise_transform_ab = scaling_transform(noise_axes_ab, noise_scales_ab)
    noise_transform_ba = scaling_transform(noise_axes_ba, noise_scales_ba)
    movement_signal_axes_ab, movement_signal_scales_ab = pooled_spectrum_scaling(
        movement_signal_a, movement_signal_b
    )
    movement_signal_axes_ba, movement_signal_scales_ba = pooled_spectrum_scaling(
        movement_signal_b, movement_signal_a
    )
    movement_residual_axes_ab, movement_residual_scales_ab = pooled_spectrum_scaling(
        movement_residual_a, movement_residual_b
    )
    movement_residual_axes_ba, movement_residual_scales_ba = pooled_spectrum_scaling(
        movement_residual_b, movement_residual_a
    )
    movement_signal_transform_ab = scaling_transform(
        movement_signal_axes_ab, movement_signal_scales_ab
    )
    movement_signal_transform_ba = scaling_transform(
        movement_signal_axes_ba, movement_signal_scales_ba
    )
    movement_residual_transform_ab = scaling_transform(
        movement_residual_axes_ab, movement_residual_scales_ab
    )
    movement_residual_transform_ba = scaling_transform(
        movement_residual_axes_ba, movement_residual_scales_ba
    )
    row.update({
        "phase_signal_scale_log_sd": float(np.std(np.log(signal_scales_ab))),
        "phase_residual_scale_log_sd": float(np.std(np.log(noise_scales_ab))),
        "movement_signal_scale_log_sd": float(
            np.std(np.log(movement_signal_scales_ab))
        ),
        "movement_residual_scale_log_sd": float(
            np.std(np.log(movement_residual_scales_ab))
        ),
    })

    observed = {}
    for mode in PHASE_MODES:
        forward, reverse = directional_transformed_scores(
            a,
            b,
            shared_a,
            shared_b,
            calibration_a,
            calibration_b,
            evaluation_a,
            evaluation_b,
            geometry_a,
            geometry_b,
            mode_transforms(signal_transform_ab, noise_transform_ab, mode),
            mode_transforms(signal_transform_ba, noise_transform_ba, mode),
        )
        selective = (forward - forward_shared) - (reverse - reverse_shared)
        row[f"fwd_{mode}"] = forward
        row[f"rev_{mode}"] = reverse
        row[f"selective_{mode}"] = selective
        observed[mode] = selective

    for mode in MOVEMENT_MODES:
        transformed_a = transform_encoded_components(
            shared_a,
            calibration_a,
            movement_prediction_a,
            *mode_transforms(
                movement_signal_transform_ab,
                movement_residual_transform_ab,
                mode,
            ),
        )
        transformed_b = transform_encoded_components(
            shared_b,
            calibration_b,
            movement_prediction_b,
            *mode_transforms(
                movement_signal_transform_ba,
                movement_residual_transform_ba,
                mode,
            ),
        )
        forward = trial_aware_kalman_score(
            a, transformed_a, calibration_a, b, shared_b, evaluation_b
        )
        reverse = trial_aware_kalman_score(
            b, transformed_b, calibration_b, a, shared_a, evaluation_a
        )
        selective = (forward - forward_shared) - (reverse - reverse_shared)
        row[f"fwd_{mode}"] = forward
        row[f"rev_{mode}"] = reverse
        row[f"selective_{mode}"] = selective
        observed[mode] = selective

    null_rows = []
    for draw in range(random_draws):
        rng = np.random.default_rng(fit_seed + 3_000_000 + draw * 10_000)
        random_signal_axes = random_orthogonal(rank, rng)
        random_signal_ab = scaling_transform(
            random_signal_axes, signal_scales_ab
        )
        random_signal_ba = scaling_transform(
            random_signal_axes, signal_scales_ba
        )
        random_noise_axes = random_orthogonal(rank, rng)
        random_noise_ab = scaling_transform(
            random_noise_axes, noise_scales_ab
        )
        random_noise_ba = scaling_transform(
            random_noise_axes, noise_scales_ba
        )
        random_movement_signal_axes = random_orthogonal(rank, rng)
        random_movement_signal_ab = scaling_transform(
            random_movement_signal_axes, movement_signal_scales_ab
        )
        random_movement_signal_ba = scaling_transform(
            random_movement_signal_axes, movement_signal_scales_ba
        )
        random_movement_residual_axes = random_orthogonal(rank, rng)
        random_movement_residual_ab = scaling_transform(
            random_movement_residual_axes, movement_residual_scales_ab
        )
        random_movement_residual_ba = scaling_transform(
            random_movement_residual_axes, movement_residual_scales_ba
        )
        for mode in PHASE_MODES:
            forward, reverse = directional_transformed_scores(
                a,
                b,
                shared_a,
                shared_b,
                calibration_a,
                calibration_b,
                evaluation_a,
                evaluation_b,
                geometry_a,
                geometry_b,
                mode_transforms(random_signal_ab, random_noise_ab, mode),
                mode_transforms(random_signal_ba, random_noise_ba, mode),
            )
            null_rows.append({
                "mode": mode,
                "draw": draw,
                "fwd_score": forward,
                "rev_score": reverse,
                "selective_change": (
                    (forward - forward_shared) - (reverse - reverse_shared)
                ),
                "observed_selective_change": observed[mode],
                "shared_rank": rank,
            })
        for mode in MOVEMENT_MODES:
            transformed_a = transform_encoded_components(
                shared_a,
                calibration_a,
                movement_prediction_a,
                *mode_transforms(
                    random_movement_signal_ab,
                    random_movement_residual_ab,
                    mode,
                ),
            )
            transformed_b = transform_encoded_components(
                shared_b,
                calibration_b,
                movement_prediction_b,
                *mode_transforms(
                    random_movement_signal_ba,
                    random_movement_residual_ba,
                    mode,
                ),
            )
            forward = trial_aware_kalman_score(
                a, transformed_a, calibration_a, b, shared_b, evaluation_b
            )
            reverse = trial_aware_kalman_score(
                b, transformed_b, calibration_b, a, shared_a, evaluation_a
            )
            null_rows.append({
                "mode": mode,
                "draw": draw,
                "fwd_score": forward,
                "rev_score": reverse,
                "selective_change": (
                    (forward - forward_shared) - (reverse - reverse_shared)
                ),
                "observed_selective_change": observed[mode],
                "shared_rank": rank,
            })
    return row, null_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-draws", type=int, default=5)
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
        f"shared signal geometry job {args.job_index + 1}/{args.num_jobs}: "
        f"{len(selected)} pairs, {len(sessions)} sessions",
        flush=True,
    )
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    rows = []
    null_rows = []
    for selected_index, (r2_index, pair_index, r1_session, r2_session) in enumerate(selected):
        a, b = cache[r1_session], cache[r2_session]
        for repeat in range(args.repeats):
            split_seed = (
                SEED + r2_index * 1_000_000 + pair_index * 10_000 + repeat * 100
            )
            folds_a = trial_folds(a["meta"], N_FOLDS, split_seed)
            folds_b = trial_folds(b["meta"], N_FOLDS, split_seed + 1)
            for fold in range(N_FOLDS):
                calibration_a, evaluation_a = matched_masks(
                    a, folds_a[fold], split_seed + fold * 20 + 5
                )
                calibration_b, evaluation_b = matched_masks(
                    b, folds_b[fold], split_seed + fold * 20 + 6
                )
                row, split_null = evaluate_split(
                    a,
                    b,
                    calibration_a,
                    calibration_b,
                    evaluation_a,
                    evaluation_b,
                    split_seed + fold,
                    args.random_draws,
                )
                identifiers = {
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "repeat": repeat,
                    "fold": fold,
                }
                row.update(identifiers)
                rows.append(row)
                for null_row in split_null:
                    null_row.update(identifiers)
                    null_rows.append(null_row)
        print(
            f"job {args.job_index + 1}: completed "
            f"{selected_index + 1}/{len(selected)} pairs",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUT_DIR / f"shared_signal_job_{args.job_index}_of_{args.num_jobs}.csv"
    null_path = OUT_DIR / f"shared_signal_null_job_{args.job_index}_of_{args.num_jobs}.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    pd.DataFrame(null_rows).to_csv(null_path, index=False)
    print(f"saved {result_path} ({len(rows)} rows)", flush=True)
    print(f"saved {null_path} ({len(null_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
