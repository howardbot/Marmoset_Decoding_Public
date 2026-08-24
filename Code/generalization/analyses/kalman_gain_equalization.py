"""Equalize original-Kalman covariance assumptions and gains across directions.

The diagnostic keeps the original concatenated-transition fit, calibration-only
PCA/CCA and held-out splits.  It asks whether direction-specific W/Q or the
resulting recursive gain sequence creates the R2->R1 minus R1->R2 contrast.
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

from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2, m2_per_trial
from decoder_model_audit import COMMON_TRIM_BINS, _score
from kalman_component_swap import source_centered_states
from kalman_components import (
    KalmanComponents,
    _gain_sequence,
    fit_kalman_components,
    predict_kalman_trials,
)
from private_readout_crossfit import (
    N_FOLDS,
    SEED,
    TARGET,
    fit_calibration_alignment,
    load_session,
    standardize_from_calibration,
    trial_folds,
)

REPO = _THIS.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "kalman_gain_equalization"
REPEATS = 5
VARIANTS = (
    "original",
    "shared_W",
    "shared_Q",
    "shared_WQ",
    "mean_gain",
    "mean_gain_shared_A",
    "mean_gain_shared_H",
    "mean_gain_shared_AH",
    "mean_gain_common_center",
    "mean_gain_common_center_shared_AH",
    "swapped_gain",
)


def _mean_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (np.asarray(first) + np.asarray(second)) / 2.0


def _covariance_model(
    model: KalmanComponents,
    other: KalmanComponents,
    share_w: bool,
    share_q: bool,
) -> KalmanComponents:
    return KalmanComponents(
        A=model.A,
        W=_mean_matrix(model.W, other.W) if share_w else model.W,
        H=model.H,
        Q=_mean_matrix(model.Q, other.Q) if share_q else model.Q,
    )


def _structure_model(
    model: KalmanComponents,
    other: KalmanComponents,
    share_a: bool,
    share_h: bool,
) -> KalmanComponents:
    return KalmanComponents(
        A=_mean_matrix(model.A, other.A) if share_a else model.A,
        W=model.W,
        H=_mean_matrix(model.H, other.H) if share_h else model.H,
        Q=model.Q,
    )


def _predict_fixed_gain_sequence(
    model: KalmanComponents,
    activity: np.ndarray,
    initial_state: np.ndarray,
    gains: list[np.ndarray],
) -> np.ndarray:
    state = np.asarray(initial_state, dtype=float).copy()
    activity = np.asarray(activity, dtype=float)
    prediction = np.empty((len(activity), len(state)), dtype=float)
    prediction[0] = state
    for time, gain in enumerate(gains):
        prior_state = model.A @ state
        state = prior_state + gain @ (activity[time + 1] - model.H @ prior_state)
        prediction[time + 1] = state
    return prediction


def _predict_trials_external_gain(
    model: KalmanComponents,
    gain_models: tuple[KalmanComponents, KalmanComponents],
    gain_weights: tuple[float, float],
    activity: np.ndarray,
    state: np.ndarray,
    meta: pd.DataFrame,
) -> np.ndarray:
    prediction = np.full_like(np.asarray(state, dtype=float), np.nan)
    cache: dict[int, list[np.ndarray]] = {}
    for indices in meta.groupby("trial_number", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        n_steps = len(indices)
        if n_steps not in cache:
            first = _gain_sequence(gain_models[0], n_steps)
            second = _gain_sequence(gain_models[1], n_steps)
            cache[n_steps] = [
                gain_weights[0] * left + gain_weights[1] * right
                for left, right in zip(first, second)
            ]
        prediction[indices] = _predict_fixed_gain_sequence(
            model,
            np.asarray(activity)[indices],
            np.asarray(state)[indices[0]],
            cache[n_steps],
        )
    return prediction


def _score_prediction(state, prediction, meta) -> tuple[float, float]:
    return (
        m2_per_trial(state, prediction, meta),
        _score(state, prediction, meta, COMMON_TRIM_BINS),
    )


def _score_model(model, activity, state, evaluation, meta) -> tuple[float, float]:
    evaluation_meta = meta[evaluation].reset_index(drop=True)
    evaluation_activity = activity[evaluation]
    evaluation_state = state[evaluation]
    prediction = predict_kalman_trials(
        model, evaluation_activity, evaluation_state, evaluation_meta
    )
    return _score_prediction(evaluation_state, prediction, evaluation_meta)


def _score_external_gain(
    model,
    gain_models,
    gain_weights,
    activity,
    state,
    evaluation,
    meta,
) -> tuple[float, float]:
    evaluation_meta = meta[evaluation].reset_index(drop=True)
    evaluation_activity = activity[evaluation]
    evaluation_state = state[evaluation]
    prediction = _predict_trials_external_gain(
        model,
        gain_models,
        gain_weights,
        evaluation_activity,
        evaluation_state,
        evaluation_meta,
    )
    return _score_prediction(evaluation_state, prediction, evaluation_meta)


def evaluate_split(
    a,
    b,
    calibration_a,
    calibration_b,
    evaluation_a,
    evaluation_b,
    fit_seed,
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
    forward = fit_kalman_components(
        activity_a[calibration_a], forward_state_a[calibration_a]
    )
    reverse = fit_kalman_components(
        activity_b[calibration_b], reverse_state_b[calibration_b]
    )
    common_center = _mean_matrix(
        a["X"][calibration_a].mean(axis=0),
        b["X"][calibration_b].mean(axis=0),
    )
    common_state_a = a["X"] - common_center
    common_state_b = b["X"] - common_center
    common_forward = fit_kalman_components(
        activity_a[calibration_a], common_state_a[calibration_a]
    )
    common_reverse = fit_kalman_components(
        activity_b[calibration_b], common_state_b[calibration_b]
    )

    rows = []
    for variant in VARIANTS:
        if variant == "original":
            forward_scores = _score_model(
                forward, activity_b, forward_state_b, evaluation_b, b["meta"]
            )
            reverse_scores = _score_model(
                reverse, activity_a, reverse_state_a, evaluation_a, a["meta"]
            )
        elif variant.startswith("shared_"):
            share_w = "W" in variant
            share_q = "Q" in variant
            forward_model = _covariance_model(forward, reverse, share_w, share_q)
            reverse_model = _covariance_model(reverse, forward, share_w, share_q)
            forward_scores = _score_model(
                forward_model, activity_b, forward_state_b, evaluation_b, b["meta"]
            )
            reverse_scores = _score_model(
                reverse_model, activity_a, reverse_state_a, evaluation_a, a["meta"]
            )
        elif variant in (
            "mean_gain_common_center",
            "mean_gain_common_center_shared_AH",
        ):
            share_structure = variant.endswith("shared_AH")
            forward_model = _structure_model(
                common_forward,
                common_reverse,
                share_a=share_structure,
                share_h=share_structure,
            )
            reverse_model = _structure_model(
                common_reverse,
                common_forward,
                share_a=share_structure,
                share_h=share_structure,
            )
            forward_scores = _score_external_gain(
                forward_model, (forward, reverse), (0.5, 0.5),
                activity_b, common_state_b, evaluation_b, b["meta"],
            )
            reverse_scores = _score_external_gain(
                reverse_model, (forward, reverse), (0.5, 0.5),
                activity_a, common_state_a, evaluation_a, a["meta"],
            )
        elif variant.startswith("mean_gain"):
            share_a = variant in ("mean_gain_shared_A", "mean_gain_shared_AH")
            share_h = variant in ("mean_gain_shared_H", "mean_gain_shared_AH")
            forward_model = _structure_model(
                forward, reverse, share_a=share_a, share_h=share_h
            )
            reverse_model = _structure_model(
                reverse, forward, share_a=share_a, share_h=share_h
            )
            forward_scores = _score_external_gain(
                forward_model, (forward, reverse), (0.5, 0.5),
                activity_b, forward_state_b, evaluation_b, b["meta"],
            )
            reverse_scores = _score_external_gain(
                reverse_model, (forward, reverse), (0.5, 0.5),
                activity_a, reverse_state_a, evaluation_a, a["meta"],
            )
        elif variant == "swapped_gain":
            forward_scores = _score_external_gain(
                forward, (reverse, forward), (1.0, 0.0),
                activity_b, forward_state_b, evaluation_b, b["meta"],
            )
            reverse_scores = _score_external_gain(
                reverse, (forward, reverse), (1.0, 0.0),
                activity_a, reverse_state_a, evaluation_a, a["meta"],
            )
        else:
            raise ValueError(f"unknown variant: {variant}")

        rows.append({
            "variant": variant,
            "fwd_native": forward_scores[0],
            "rev_native": reverse_scores[0],
            "gap_native": reverse_scores[0] - forward_scores[0],
            "fwd_common": forward_scores[1],
            "rev_common": reverse_scores[1],
            "gap_common": reverse_scores[1] - forward_scores[1],
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-index", type=int, required=True, choices=range(len(SESSIONS_R2)))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r2_session = SESSIONS_R2[args.r2_index]
    pairs = list(enumerate(SESSIONS_R1))
    suffix = ""
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
        suffix = "_smoke"
    sessions = [session for _, session in pairs] + [r2_session]
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    rows = []
    for completed, (pair_index, r1_session) in enumerate(pairs, start=1):
        a, b = cache[r1_session], cache[r2_session]
        for repeat in range(args.repeats):
            split_seed = (
                SEED + args.r2_index * 1_000_000 + pair_index * 10_000 + repeat * 100
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
            f"gain equalization R2 {args.r2_index}: {completed}/{len(pairs)} pairs",
            flush=True,
        )
    output = OUT_DIR / f"gain_equalization_r2_{args.r2_index}{suffix}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
