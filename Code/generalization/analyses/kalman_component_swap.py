"""Repeated cross-fitted A/W/H/Q swaps for the Kalman transfer asymmetry."""
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
from kalman_components import (
    fit_kalman_components,
    hybrid_components,
    mask_label,
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
OUT_DIR = REPO / "Results" / "manifold_geometry" / "kalman_component_swap"
TRANSITION_MODES = ("concatenated", "trial_aware")
N_COMPONENT_MASKS = 16


def source_centered_states(source, target, source_calibration):
    """Express both days in the source decoder's native state coordinates."""
    center = source["X"][source_calibration].mean(axis=0)
    return source["X"] - center, target["X"] - center


def fit_day_components(activity, state, calibration, meta, transition_mode):
    calibration_meta = meta[calibration].reset_index(drop=True)
    transition_meta = calibration_meta if transition_mode == "trial_aware" else None
    return fit_kalman_components(
        activity[calibration], state[calibration], transition_meta
    )


def score_model(model, activity, state, evaluation, meta):
    evaluation_meta = meta[evaluation].reset_index(drop=True)
    evaluation_activity = activity[evaluation]
    evaluation_state = state[evaluation]
    prediction = predict_kalman_trials(
        model, evaluation_activity, evaluation_state, evaluation_meta
    )
    return m2_per_trial(evaluation_state, prediction, evaluation_meta)


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

    rows = []
    for transition_mode in TRANSITION_MODES:
        forward_source = fit_day_components(
            activity_a, forward_state_a, calibration_a, a["meta"], transition_mode
        )
        forward_target = fit_day_components(
            activity_b, forward_state_b, calibration_b, b["meta"], transition_mode
        )
        reverse_source = fit_day_components(
            activity_b, reverse_state_b, calibration_b, b["meta"], transition_mode
        )
        reverse_target = fit_day_components(
            activity_a, reverse_state_a, calibration_a, a["meta"], transition_mode
        )
        for mask in range(N_COMPONENT_MASKS):
            forward_model = hybrid_components(forward_source, forward_target, mask)
            reverse_model = hybrid_components(reverse_source, reverse_target, mask)
            rows.append({
                "transition_mode": transition_mode,
                "target_mask": mask,
                "target_components": mask_label(mask),
                "fwd_score": score_model(
                    forward_model,
                    activity_b,
                    forward_state_b,
                    evaluation_b,
                    b["meta"],
                ),
                "rev_score": score_model(
                    reverse_model,
                    activity_a,
                    reverse_state_a,
                    evaluation_a,
                    a["meta"],
                ),
            })
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--r2-index", type=int, choices=range(len(SESSIONS_R2)))
    group.add_argument("--job-index", type=int)
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def selected_pairs(args):
    pairs = [
        (r2_index, pair_index, r1_session, r2_session)
        for r2_index, r2_session in enumerate(SESSIONS_R2)
        for pair_index, r1_session in enumerate(SESSIONS_R1)
    ]
    if args.r2_index is not None:
        selected = [pair for pair in pairs if pair[0] == args.r2_index]
        output = OUT_DIR / f"component_swap_shard_{args.r2_index}.csv"
        label = f"R2 shard {args.r2_index}"
    else:
        if not 0 <= args.job_index < args.num_jobs:
            raise ValueError("job-index must be in [0, num-jobs)")
        selected = [
            pair for index, pair in enumerate(pairs)
            if index % args.num_jobs == args.job_index
        ]
        output = OUT_DIR / f"component_swap_job_{args.job_index}_of_{args.num_jobs}.csv"
        label = f"job {args.job_index + 1}/{args.num_jobs}"
    if args.max_pairs is not None:
        selected = selected[:args.max_pairs]
        output = output.with_name(f"{output.stem}_smoke.csv")
    return selected, output, label


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs, output, label = selected_pairs(args)
    sessions = sorted({
        session
        for _, _, r1_session, r2_session in pairs
        for session in (r1_session, r2_session)
    })
    print(
        f"loading component swap {label}: {len(pairs)} pairs, {len(sessions)} sessions",
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
            f"component swap {label}: completed {completed}/{len(pairs)} pairs",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
