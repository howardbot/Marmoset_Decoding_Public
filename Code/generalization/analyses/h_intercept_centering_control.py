"""Behaviour-only centering falsifier for the affine Kalman observation offset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

from big_sweep_phase2_crossday import ANIMAL_SESSIONS, EXCLUDE_TRIALS
from h_observation_decomposition import centering_offset
from h_observation_fine_swap import observation_families, score_observation
from kalman_component_swap import fit_day_components, source_centered_states
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
OUT_DIR = REPO / "Results" / "manifold_geometry" / "h_observation_fine_swap"
Q_CONTEXTS = ("source", "target")


def append_direction_rows(
    rows,
    direction,
    source_components,
    concatenated_components,
    source_observation,
    target_observation,
    target_activity,
    target_state,
    target_calibration,
    target_evaluation,
    target_meta,
):
    target_calibration_state = target_state[target_calibration]
    behaviour_offset = centering_offset(
        source_observation.H, target_calibration_state
    )
    target_identity_offset = centering_offset(
        target_observation.H, target_calibration_state
    )
    identity_error = float(np.max(np.abs(
        target_observation.b - target_identity_offset
    )))
    if identity_error > 1e-10:
        raise AssertionError(f"target affine-centering identity failed: {identity_error}")
    conditions = (
        ("source_b", source_observation.b),
        ("behaviour_center", behaviour_offset),
        ("target_b", target_observation.b),
    )
    rows.append({
        "direction": direction,
        "q_context": "source",
        "condition": "original_trial_aware",
        "target_identity_error": identity_error,
        "offset_norm": 0.0,
        "score": score_observation(
            source_components,
            source_components.H,
            source_components.Q,
            np.zeros(source_components.H.shape[0]),
            target_activity,
            target_state,
            target_evaluation,
            target_meta,
        ),
    })
    rows.append({
        "direction": direction,
        "q_context": "source",
        "condition": "original_concatenated",
        "target_identity_error": identity_error,
        "offset_norm": 0.0,
        "score": score_observation(
            concatenated_components,
            concatenated_components.H,
            concatenated_components.Q,
            np.zeros(concatenated_components.H.shape[0]),
            target_activity,
            target_state,
            target_evaluation,
            target_meta,
        ),
    })
    for q_context in Q_CONTEXTS:
        Q = (
            source_observation.Q if q_context == "source"
            else target_observation.Q
        )
        for condition, b in conditions:
            rows.append({
                "direction": direction,
                "q_context": q_context,
                "condition": condition,
                "target_identity_error": identity_error,
                "offset_norm": float(np.linalg.norm(b)),
                "score": score_observation(
                    source_components,
                    source_observation.H,
                    Q,
                    b,
                    target_activity,
                    target_state,
                    target_evaluation,
                    target_meta,
                ),
            })


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
    forward_source = fit_day_components(
        activity_a, forward_state_a, calibration_a, a["meta"], "trial_aware"
    )
    forward_source_concatenated = fit_day_components(
        activity_a, forward_state_a, calibration_a, a["meta"], "concatenated"
    )
    forward_target = fit_day_components(
        activity_b, forward_state_b, calibration_b, b["meta"], "trial_aware"
    )
    reverse_source = fit_day_components(
        activity_b, reverse_state_b, calibration_b, b["meta"], "trial_aware"
    )
    reverse_source_concatenated = fit_day_components(
        activity_b, reverse_state_b, calibration_b, b["meta"], "concatenated"
    )
    reverse_target = fit_day_components(
        activity_a, reverse_state_a, calibration_a, a["meta"], "trial_aware"
    )
    forward_source_observation = observation_families(
        forward_source, activity_a, forward_state_a, calibration_a
    )["affine"]
    forward_target_observation = observation_families(
        forward_target, activity_b, forward_state_b, calibration_b
    )["affine"]
    reverse_source_observation = observation_families(
        reverse_source, activity_b, reverse_state_b, calibration_b
    )["affine"]
    reverse_target_observation = observation_families(
        reverse_target, activity_a, reverse_state_a, calibration_a
    )["affine"]
    rows = []
    append_direction_rows(
        rows,
        "forward",
        forward_source,
        forward_source_concatenated,
        forward_source_observation,
        forward_target_observation,
        activity_b,
        forward_state_b,
        calibration_b,
        evaluation_b,
        b["meta"],
    )
    append_direction_rows(
        rows,
        "reverse",
        reverse_source,
        reverse_source_concatenated,
        reverse_source_observation,
        reverse_target_observation,
        activity_a,
        reverse_state_a,
        calibration_a,
        evaluation_a,
        a["meta"],
    )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=sorted(ANIMAL_SESSIONS), default="TS")
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.job_index < args.num_jobs:
        raise ValueError("job-index must be in [0, num-jobs)")
    sessions_r1, sessions_r2 = ANIMAL_SESSIONS[args.animal]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = [
        (r2_index, pair_index, r1_session, r2_session)
        for r2_index, r2_session in enumerate(sessions_r2)
        for pair_index, r1_session in enumerate(sessions_r1)
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
        f"loading H-centering animal={args.animal} "
        f"job {args.job_index + 1}/{args.num_jobs}: "
        f"{len(pairs)} pairs",
        flush=True,
    )
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    rows = []
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
                    "animal": args.animal,
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
            f"H-centering animal={args.animal} "
            f"job {args.job_index + 1}/{args.num_jobs}: "
            f"completed {completed}/{len(pairs)}",
            flush=True,
        )
    suffix = f"job_{args.job_index}_of_{args.num_jobs}"
    if args.animal != "TS":
        suffix = f"{args.animal.lower()}_{suffix}"
    if args.max_pairs is not None:
        suffix += "_smoke"
    output = OUT_DIR / f"h_centering_{suffix}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
