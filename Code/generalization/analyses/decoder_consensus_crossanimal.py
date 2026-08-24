"""Matched-split cross-animal decoder consensus and source-quality audit.

All decoder families use the same calibration-only PCA/CCA alignment, the
same held-out whole trials, and a common scoring support.  The primary
consensus includes ridge, trial-aware Wiener, trial-aware ARX, and trial-aware
Kalman.  The legacy concatenated Kalman and calibration-only behavior-centered
Kalman are retained as mechanism controls, but are not counted as additional
independent decoder families.

For every transfer direction the source model is also scored on its own
held-out trials.  This permits a direct test of whether a raw cross-day gap is
merely inherited from unequal source-decoder quality.
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

from big_sweep_phase2_crossday import ANIMAL_SESSIONS, EXCLUDE_TRIALS
from decoder_model_audit import (
    COMMON_TRIM_BINS,
    REPEATS,
    RIDGE_ALPHA,
    TARGETS,
    WIENER_HISTORY_BINS,
    _fit_linear,
    _score,
    predict_arx_trials,
    score_ridge,
    score_wiener,
)
from h_observation_decomposition import centering_offset
from kalman_component_swap import source_centered_states
from kalman_components import (
    KalmanComponents,
    fit_kalman_components,
    predict_kalman_trials,
    transition_indices,
)
from private_readout_crossfit import (
    N_FOLDS,
    SEED,
    fit_calibration_alignment,
    load_session,
    standardize_from_calibration,
    trial_folds,
)


REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "decoder_consensus_crossanimal"
PRIMARY_CONDITIONS = (
    ("ridge", "instantaneous"),
    ("wiener", f"history_{WIENER_HISTORY_BINS}"),
    ("arx", "trial_aware"),
    ("kalman", "trial_aware"),
)


def fit_arx_trial_aware(
    activity: np.ndarray,
    state: np.ndarray,
    calibration: np.ndarray,
    meta: pd.DataFrame,
) -> np.ndarray:
    """Fit recursive ARX coefficients without transitions across trials."""
    calibration_activity = np.asarray(activity)[calibration]
    calibration_state = np.asarray(state)[calibration]
    calibration_meta = meta[calibration].reset_index(drop=True)
    source, target = transition_indices(calibration_meta, len(calibration_meta))
    features = np.column_stack([
        calibration_state[source], calibration_activity[target]
    ])
    return _fit_linear(features, calibration_state[target], RIDGE_ALPHA)


def score_arx_model(
    coefficients: np.ndarray,
    activity: np.ndarray,
    state: np.ndarray,
    evaluation: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[float, float]:
    evaluation_meta = meta[evaluation].reset_index(drop=True)
    values = np.asarray(state)[evaluation]
    prediction = predict_arx_trials(
        coefficients,
        np.asarray(activity)[evaluation],
        values,
        evaluation_meta,
    )
    return (
        _score(values, prediction, evaluation_meta),
        _score(values, prediction, evaluation_meta, COMMON_TRIM_BINS),
    )


def score_kalman_offset(
    model: KalmanComponents,
    offset: np.ndarray,
    activity: np.ndarray,
    state: np.ndarray,
    evaluation: np.ndarray,
    meta: pd.DataFrame,
) -> tuple[float, float]:
    """Score a Kalman observation model after subtracting an affine offset."""
    evaluation_meta = meta[evaluation].reset_index(drop=True)
    values = np.asarray(state)[evaluation]
    observations = np.asarray(activity)[evaluation] - np.asarray(offset)[None, :]
    prediction = predict_kalman_trials(
        model, observations, values, evaluation_meta
    )
    return (
        _score(values, prediction, evaluation_meta),
        _score(values, prediction, evaluation_meta, COMMON_TRIM_BINS),
    )


def result_row(
    decoder: str,
    variant: str,
    consensus_included: bool,
    forward_cross: tuple[float, float],
    reverse_cross: tuple[float, float],
    forward_own: tuple[float, float],
    reverse_own: tuple[float, float],
) -> dict:
    return {
        "decoder": decoder,
        "variant": variant,
        "consensus_included": bool(consensus_included),
        "fwd_cross_native": forward_cross[0],
        "rev_cross_native": reverse_cross[0],
        "fwd_cross_common": forward_cross[1],
        "rev_cross_common": reverse_cross[1],
        "fwd_own_native": forward_own[0],
        "rev_own_native": reverse_own[0],
        "fwd_own_common": forward_own[1],
        "rev_own_common": reverse_own[1],
    }


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
    rows = []

    forward_args = (
        activity_a, forward_state_a, calibration_a,
        activity_b, forward_state_b, evaluation_b, b["meta"],
    )
    reverse_args = (
        activity_b, reverse_state_b, calibration_b,
        activity_a, reverse_state_a, evaluation_a, a["meta"],
    )
    forward_own_args = (
        activity_a, forward_state_a, calibration_a,
        activity_a, forward_state_a, evaluation_a, a["meta"],
    )
    reverse_own_args = (
        activity_b, reverse_state_b, calibration_b,
        activity_b, reverse_state_b, evaluation_b, b["meta"],
    )
    rows.append(result_row(
        "ridge", "instantaneous", True,
        score_ridge(*forward_args), score_ridge(*reverse_args),
        score_ridge(*forward_own_args), score_ridge(*reverse_own_args),
    ))

    def wiener_args(source_activity, source_state, source_calibration, source_meta,
                    target_activity, target_state, target_evaluation, target_meta):
        return (
            source_activity, source_state, source_calibration, source_meta,
            target_activity, target_state, target_evaluation, target_meta,
            WIENER_HISTORY_BINS,
        )

    rows.append(result_row(
        "wiener", f"history_{WIENER_HISTORY_BINS}", True,
        score_wiener(*wiener_args(
            activity_a, forward_state_a, calibration_a, a["meta"],
            activity_b, forward_state_b, evaluation_b, b["meta"],
        )),
        score_wiener(*wiener_args(
            activity_b, reverse_state_b, calibration_b, b["meta"],
            activity_a, reverse_state_a, evaluation_a, a["meta"],
        )),
        score_wiener(*wiener_args(
            activity_a, forward_state_a, calibration_a, a["meta"],
            activity_a, forward_state_a, evaluation_a, a["meta"],
        )),
        score_wiener(*wiener_args(
            activity_b, reverse_state_b, calibration_b, b["meta"],
            activity_b, reverse_state_b, evaluation_b, b["meta"],
        )),
    ))

    forward_arx = fit_arx_trial_aware(
        activity_a, forward_state_a, calibration_a, a["meta"]
    )
    reverse_arx = fit_arx_trial_aware(
        activity_b, reverse_state_b, calibration_b, b["meta"]
    )
    rows.append(result_row(
        "arx", "trial_aware", True,
        score_arx_model(
            forward_arx, activity_b, forward_state_b, evaluation_b, b["meta"]
        ),
        score_arx_model(
            reverse_arx, activity_a, reverse_state_a, evaluation_a, a["meta"]
        ),
        score_arx_model(
            forward_arx, activity_a, forward_state_a, evaluation_a, a["meta"]
        ),
        score_arx_model(
            reverse_arx, activity_b, reverse_state_b, evaluation_b, b["meta"]
        ),
    ))

    calibration_meta_a = a["meta"][calibration_a].reset_index(drop=True)
    calibration_meta_b = b["meta"][calibration_b].reset_index(drop=True)
    forward_kalman_concatenated = fit_kalman_components(
        activity_a[calibration_a], forward_state_a[calibration_a]
    )
    reverse_kalman_concatenated = fit_kalman_components(
        activity_b[calibration_b], reverse_state_b[calibration_b]
    )
    forward_kalman = fit_kalman_components(
        activity_a[calibration_a], forward_state_a[calibration_a],
        calibration_meta_a,
    )
    reverse_kalman = fit_kalman_components(
        activity_b[calibration_b], reverse_state_b[calibration_b],
        calibration_meta_b,
    )
    zero = np.zeros(activity_a.shape[1])
    forward_own_original = score_kalman_offset(
        forward_kalman_concatenated, zero, activity_a, forward_state_a,
        evaluation_a, a["meta"],
    )
    reverse_own_original = score_kalman_offset(
        reverse_kalman_concatenated, zero, activity_b, reverse_state_b,
        evaluation_b, b["meta"],
    )
    rows.append(result_row(
        "kalman", "original_concatenated", False,
        score_kalman_offset(
            forward_kalman_concatenated, zero, activity_b, forward_state_b,
            evaluation_b, b["meta"],
        ),
        score_kalman_offset(
            reverse_kalman_concatenated, zero, activity_a, reverse_state_a,
            evaluation_a, a["meta"],
        ),
        forward_own_original,
        reverse_own_original,
    ))
    forward_own_trial = score_kalman_offset(
        forward_kalman, zero, activity_a, forward_state_a,
        evaluation_a, a["meta"],
    )
    reverse_own_trial = score_kalman_offset(
        reverse_kalman, zero, activity_b, reverse_state_b,
        evaluation_b, b["meta"],
    )
    rows.append(result_row(
        "kalman", "trial_aware", True,
        score_kalman_offset(
            forward_kalman, zero, activity_b, forward_state_b,
            evaluation_b, b["meta"],
        ),
        score_kalman_offset(
            reverse_kalman, zero, activity_a, reverse_state_a,
            evaluation_a, a["meta"],
        ),
        forward_own_trial,
        reverse_own_trial,
    ))
    forward_offset = centering_offset(
        forward_kalman.H, forward_state_b[calibration_b]
    )
    reverse_offset = centering_offset(
        reverse_kalman.H, reverse_state_a[calibration_a]
    )
    rows.append(result_row(
        "kalman", "behaviour_center", False,
        score_kalman_offset(
            forward_kalman, forward_offset, activity_b, forward_state_b,
            evaluation_b, b["meta"],
        ),
        score_kalman_offset(
            reverse_kalman, reverse_offset, activity_a, reverse_state_a,
            evaluation_a, a["meta"],
        ),
        forward_own_trial,
        reverse_own_trial,
    ))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=sorted(ANIMAL_SESSIONS), default="TS")
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--r2-index", required=True, type=int)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = ANIMAL_SESSIONS[args.animal]
    if not 0 <= args.r2_index < len(sessions_r2):
        raise ValueError(
            f"r2-index must be in [0, {len(sessions_r2)}) for {args.animal}"
        )
    r2_session = sessions_r2[args.r2_index]
    r1_sessions = sessions_r1[: args.max_pairs]
    sessions = list(r1_sessions) + [r2_session]
    print(
        f"decoder consensus animal={args.animal} target={args.target} "
        f"R2[{args.r2_index}] pairs={len(r1_sessions)} repeats={args.repeats}",
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
                SEED + args.r2_index * 1_000_000
                + pair_index * 10_000 + repeat * 100
            )
            folds_a = trial_folds(a["meta"], N_FOLDS, split_seed)
            folds_b = trial_folds(b["meta"], N_FOLDS, split_seed + 1)
            for fold in range(N_FOLDS):
                evaluation_a = folds_a[fold]
                evaluation_b = folds_b[fold]
                split_rows = evaluate_split(
                    a, b, ~evaluation_a, ~evaluation_b,
                    evaluation_a, evaluation_b, split_seed + fold,
                )
                identifiers = {
                    "animal": args.animal,
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
            f"decoder consensus {args.animal} {args.target} "
            f"R2[{args.r2_index}]: {pair_index + 1}/{len(r1_sessions)} pairs",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.max_pairs is not None or args.repeats != REPEATS:
        suffix = "_smoke"
    output = OUT_DIR / (
        f"consensus_{args.animal.lower()}_{args.target}_"
        f"r2_{args.r2_index}{suffix}.csv"
    )
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
