"""Matched-split audit of decoder dependence in the R1/R2 directional gap.

Every decoder sees the same calibration-only PCA/CCA alignment and the same
held-out trials.  The audit includes the original concatenated-transition
Kalman, feed-forward ridge and Wiener decoders, and an independent recursive
ARX decoder.  Each model uses one parameter set selected without reference to
the cross-day directional gap.

The script is shardable by target and R2 day.  It deliberately uses the
original concatenated-transition convention; no trial-aware fits are run.
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

import decoder_utils as du
from Neural_Decoding.decoders import WienerFilterRegression
from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2, m2_per_trial
from kalman_component_swap import source_centered_states
from kalman_components import fit_kalman_components, predict_kalman_trials
from private_readout_crossfit import (
    N_FOLDS,
    SEED,
    fit_calibration_alignment,
    load_session,
    standardize_from_calibration,
    trial_folds,
)

REPO = _THIS.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "decoder_audit"
TARGETS = ("relative_position", "relative_velocity")
REPEATS = 5
RIDGE_ALPHA = 1e-2
# The existing same-session sweep selects the 50 ms Wiener setting at the
# locked 30 ms bin size (two prior bins).  Parameters are fixed before this
# cross-day audit; no cross-day gap is used for model selection.
WIENER_HISTORY_BINS = 2
COMMON_TRIM_BINS = WIENER_HISTORY_BINS


def _fit_linear(features: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    """Fit multivariate ridge with an unpenalized intercept."""
    design = np.column_stack([np.ones(len(features)), features])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + penalty,
        design.T @ targets,
    )


def _predict_linear(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features)), features]) @ coefficients


def _trim_trials(values, prediction, meta, n_bins: int):
    """Remove the first ``n_bins`` rows of every trial."""
    if n_bins <= 0:
        return values, prediction, meta
    keep = np.zeros(len(meta), dtype=bool)
    for indices in meta.groupby("trial_number", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        keep[indices[n_bins:]] = True
    return values[keep], prediction[keep], meta[keep].reset_index(drop=True)


def _score(values, prediction, meta, trim_bins: int = 0) -> float:
    values, prediction, meta = _trim_trials(
        np.asarray(values), np.asarray(prediction), meta, trim_bins
    )
    return m2_per_trial(values, prediction, meta)


def score_kalman(model, activity, state, evaluation, meta) -> tuple[float, float]:
    evaluation_meta = meta[evaluation].reset_index(drop=True)
    evaluation_activity = activity[evaluation]
    evaluation_state = state[evaluation]
    prediction = predict_kalman_trials(
        model, evaluation_activity, evaluation_state, evaluation_meta
    )
    return (
        _score(evaluation_state, prediction, evaluation_meta),
        _score(evaluation_state, prediction, evaluation_meta, COMMON_TRIM_BINS),
    )


def score_ridge(
    source_activity,
    source_state,
    source_calibration,
    target_activity,
    target_state,
    target_evaluation,
    target_meta,
) -> tuple[float, float]:
    coefficients = _fit_linear(
        source_activity[source_calibration],
        source_state[source_calibration],
        RIDGE_ALPHA,
    )
    evaluation_meta = target_meta[target_evaluation].reset_index(drop=True)
    values = target_state[target_evaluation]
    prediction = _predict_linear(target_activity[target_evaluation], coefficients)
    return (
        _score(values, prediction, evaluation_meta),
        _score(values, prediction, evaluation_meta, COMMON_TRIM_BINS),
    )


def score_wiener(
    source_activity,
    source_state,
    source_calibration,
    source_meta,
    target_activity,
    target_state,
    target_evaluation,
    target_meta,
    history_bins: int,
) -> tuple[float, float]:
    calibration_meta = source_meta[source_calibration].reset_index(drop=True)
    evaluation_meta = target_meta[target_evaluation].reset_index(drop=True)
    train_features, train_values, _ = du.make_history_features(
        source_state[source_calibration],
        source_activity[source_calibration],
        calibration_meta,
        history_bins,
        lag_bins=0,
    )
    test_features, test_values, test_meta = du.make_history_features(
        target_state[target_evaluation],
        target_activity[target_evaluation],
        evaluation_meta,
        history_bins,
        lag_bins=0,
    )
    feature_mean = train_features.mean(axis=0)
    value_mean = train_values.mean(axis=0)
    model = WienerFilterRegression()
    model.fit(train_features - feature_mean, train_values - value_mean)
    prediction = model.predict(test_features - feature_mean) + value_mean
    return (
        _score(test_values, prediction, test_meta),
        _score(
            test_values,
            prediction,
            test_meta,
            COMMON_TRIM_BINS - history_bins,
        ),
    )


def fit_arx(activity: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Fit x[t] from recursively predicted x[t-1] and observed z[t]."""
    features = np.column_stack([state[:-1], activity[1:]])
    return _fit_linear(features, state[1:], RIDGE_ALPHA)


def predict_arx_trials(coefficients, activity, state, meta) -> np.ndarray:
    prediction = np.full_like(state, np.nan, dtype=float)
    for indices in meta.groupby("trial_number", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        prediction[indices[0]] = state[indices[0]]
        for index in indices[1:]:
            features = np.concatenate([prediction[index - 1], activity[index]])
            prediction[index] = _predict_linear(features[None, :], coefficients)[0]
    return prediction


def score_arx(
    source_activity,
    source_state,
    source_calibration,
    target_activity,
    target_state,
    target_evaluation,
    target_meta,
) -> tuple[float, float]:
    coefficients = fit_arx(
        source_activity[source_calibration], source_state[source_calibration]
    )
    evaluation_meta = target_meta[target_evaluation].reset_index(drop=True)
    values = target_state[target_evaluation]
    prediction = predict_arx_trials(
        coefficients, target_activity[target_evaluation], values, evaluation_meta
    )
    return (
        _score(values, prediction, evaluation_meta),
        _score(values, prediction, evaluation_meta, COMMON_TRIM_BINS),
    )


def evaluate_split(
    a,
    b,
    calibration_a,
    calibration_b,
    evaluation_a,
    evaluation_b,
    fit_seed: int,
) -> list[dict]:
    alignment = fit_calibration_alignment(
        a, b, calibration_a, calibration_b, fit_seed
    )
    activity_a = standardize_from_calibration(
        alignment.transform_train(a["Y"]), calibration_a
    )
    activity_b = standardize_from_calibration(
        alignment.transform_target(b["Y"]), calibration_b
    )
    forward_state_a, forward_state_b = source_centered_states(
        a, b, calibration_a
    )
    reverse_state_b, reverse_state_a = source_centered_states(
        b, a, calibration_b
    )

    forward_source = fit_kalman_components(
        activity_a[calibration_a], forward_state_a[calibration_a]
    )
    reverse_source = fit_kalman_components(
        activity_b[calibration_b], reverse_state_b[calibration_b]
    )

    rows = []
    forward_native, forward_common = score_kalman(
        forward_source,
        activity_b,
        forward_state_b,
        evaluation_b,
        b["meta"],
    )
    reverse_native, reverse_common = score_kalman(
        reverse_source,
        activity_a,
        reverse_state_a,
        evaluation_a,
        a["meta"],
    )
    rows.append({
        "decoder": "kalman",
        "variant": "original",
        "history_bins": np.nan,
        "kalman_C": 1.0,
        "fwd_native": forward_native,
        "rev_native": reverse_native,
        "fwd_common": forward_common,
        "rev_common": reverse_common,
    })

    direction_args = (
        activity_a,
        forward_state_a,
        calibration_a,
        activity_b,
        forward_state_b,
        evaluation_b,
        b["meta"],
    )
    reverse_args = (
        activity_b,
        reverse_state_b,
        calibration_b,
        activity_a,
        reverse_state_a,
        evaluation_a,
        a["meta"],
    )
    for decoder, scorer in (("ridge", score_ridge), ("arx", score_arx)):
        forward_native, forward_common = scorer(*direction_args)
        reverse_native, reverse_common = scorer(*reverse_args)
        rows.append({
            "decoder": decoder,
            "variant": "original",
            "history_bins": 0,
            "kalman_C": np.nan,
            "fwd_native": forward_native,
            "rev_native": reverse_native,
            "fwd_common": forward_common,
            "rev_common": reverse_common,
        })

    history_bins = WIENER_HISTORY_BINS
    forward_native, forward_common = score_wiener(
        activity_a,
        forward_state_a,
        calibration_a,
        a["meta"],
        activity_b,
        forward_state_b,
        evaluation_b,
        b["meta"],
        history_bins,
    )
    reverse_native, reverse_common = score_wiener(
        activity_b,
        reverse_state_b,
        calibration_b,
        b["meta"],
        activity_a,
        reverse_state_a,
        evaluation_a,
        a["meta"],
        history_bins,
    )
    rows.append({
        "decoder": "wiener",
        "variant": f"history_{history_bins}",
        "history_bins": history_bins,
        "kalman_C": np.nan,
        "fwd_native": forward_native,
        "rev_native": reverse_native,
        "fwd_common": forward_common,
        "rev_common": reverse_common,
    })
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument(
        "--r2-index", required=True, type=int, choices=range(len(SESSIONS_R2))
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r2_session = SESSIONS_R2[args.r2_index]
    r1_sessions = SESSIONS_R1[: args.max_pairs]
    sessions = list(r1_sessions) + [r2_session]
    print(
        f"decoder audit target={args.target} R2[{args.r2_index}] "
        f"pairs={len(r1_sessions)} repeats={args.repeats}",
        flush=True,
    )
    cache = {
        session: load_session(
            session, args.target, EXCLUDE_TRIALS.get(session, ())
        )
        for session in sessions
    }

    rows = []
    for pair_index, r1_session in enumerate(r1_sessions):
        a, b = cache[r1_session], cache[r2_session]
        for repeat in range(args.repeats):
            split_seed = (
                SEED
                + args.r2_index * 1_000_000
                + pair_index * 10_000
                + repeat * 100
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
                    "target": args.target,
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "repeat": repeat,
                    "fold": fold,
                }
                for row in split_rows:
                    row.update(identifiers)
                    rows.append(row)
        print(
            f"decoder audit {args.target} R2[{args.r2_index}]: "
            f"{pair_index + 1}/{len(r1_sessions)} pairs",
            flush=True,
        )

    suffix = "_smoke" if args.max_pairs is not None or args.repeats != REPEATS else ""
    output = OUT_DIR / f"audit_{args.target}_r2_{args.r2_index}{suffix}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
