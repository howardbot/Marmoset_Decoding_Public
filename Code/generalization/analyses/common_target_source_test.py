"""Identify source-side transfer asymmetry using a fixed third-session target.

For each R1/R2 source pair, both decoders predict the exact same held-out trials
from a third session.  Calibration and evaluation trial counts are fixed, and
the target is never either source session.  This removes target-day difficulty
from the R2-source minus R1-source contrast.
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
from decoder_consensus_crossanimal import (
    PRIMARY_CONDITIONS,
    fit_arx_trial_aware,
    score_arx_model,
    score_kalman_offset,
)
from decoder_model_audit import (
    REPEATS,
    TARGETS,
    WIENER_HISTORY_BINS,
    score_ridge,
    score_wiener,
)
from h_observation_decomposition import centering_offset
from kalman_components import fit_kalman_components
from private_readout_crossfit import (
    N_FOLDS,
    SEED,
    fit_calibration_alignment,
    load_session,
    standardize_from_calibration,
    trial_folds,
)
from readout_compactness_crossfit import select_trials


N_SOURCE_CALIBRATION_TRIALS = 32
N_TARGET_CALIBRATION_TRIALS = 32
N_TARGET_EVALUATION_TRIALS = 8
CONDITIONS = (
    ("ridge", "instantaneous"),
    ("wiener", f"history_{WIENER_HISTORY_BINS}"),
    ("arx", "trial_aware"),
    ("kalman", "original_concatenated"),
    ("kalman", "trial_aware"),
    ("kalman", "behaviour_center"),
)
REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "common_target_source_test"


def select_common_target(
    sessions: tuple[str, ...] | list[str],
    excluded: str,
    selection_index: int,
) -> str:
    candidates = [session for session in sessions if session != excluded]
    if not candidates:
        raise ValueError("a common target requires a session distinct from the source")
    return candidates[selection_index % len(candidates)]


def score_source_to_target(
    source,
    target,
    source_calibration: np.ndarray,
    target_calibration: np.ndarray,
    target_evaluation: np.ndarray,
    fit_seed: int,
) -> dict[tuple[str, str], float]:
    alignment = fit_calibration_alignment(
        source, target, source_calibration, target_calibration, fit_seed
    )
    source_activity = standardize_from_calibration(
        alignment.transform_train(source["Y"]), source_calibration
    )
    target_activity = standardize_from_calibration(
        alignment.transform_target(target["Y"]), target_calibration
    )
    center = source["X"][source_calibration].mean(axis=0)
    source_state = source["X"] - center
    target_state = target["X"] - center
    scores = {}
    ridge_args = (
        source_activity, source_state, source_calibration,
        target_activity, target_state, target_evaluation, target["meta"],
    )
    scores[("ridge", "instantaneous")] = score_ridge(*ridge_args)[1]
    scores[("wiener", f"history_{WIENER_HISTORY_BINS}")] = score_wiener(
        source_activity, source_state, source_calibration, source["meta"],
        target_activity, target_state, target_evaluation, target["meta"],
        WIENER_HISTORY_BINS,
    )[1]
    arx = fit_arx_trial_aware(
        source_activity, source_state, source_calibration, source["meta"]
    )
    scores[("arx", "trial_aware")] = score_arx_model(
        arx, target_activity, target_state, target_evaluation, target["meta"]
    )[1]

    calibration_meta = source["meta"][source_calibration].reset_index(drop=True)
    concatenated = fit_kalman_components(
        source_activity[source_calibration], source_state[source_calibration]
    )
    trial_aware = fit_kalman_components(
        source_activity[source_calibration], source_state[source_calibration],
        calibration_meta,
    )
    zero = np.zeros(source_activity.shape[1])
    scores[("kalman", "original_concatenated")] = score_kalman_offset(
        concatenated, zero, target_activity, target_state,
        target_evaluation, target["meta"],
    )[1]
    scores[("kalman", "trial_aware")] = score_kalman_offset(
        trial_aware, zero, target_activity, target_state,
        target_evaluation, target["meta"],
    )[1]
    offset = centering_offset(
        trial_aware.H, target_state[target_calibration]
    )
    scores[("kalman", "behaviour_center")] = score_kalman_offset(
        trial_aware, offset, target_activity, target_state,
        target_evaluation, target["meta"],
    )[1]
    if set(scores) != set(CONDITIONS):
        raise AssertionError("common-target decoder conditions are incomplete")
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=sorted(ANIMAL_SESSIONS), required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--target-epoch", choices=("R1", "R2"), required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--num-jobs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-triads", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.job_index < args.num_jobs:
        raise ValueError("job-index must be in [0, num-jobs)")
    sessions_r1, sessions_r2 = ANIMAL_SESSIONS[args.animal]
    if args.target_epoch == "R2" and len(sessions_r2) < 2:
        raise ValueError(f"{args.animal} has no independent R2 common target")
    all_triads = []
    for r2_index, r2_source in enumerate(sessions_r2):
        for r1_index, r1_source in enumerate(sessions_r1):
            target_sessions = sessions_r1 if args.target_epoch == "R1" else sessions_r2
            excluded = r1_source if args.target_epoch == "R1" else r2_source
            common_target = select_common_target(
                target_sessions, excluded, r1_index + r2_index
            )
            all_triads.append((
                r1_index, r2_index, r1_source, r2_source, common_target
            ))
    triads = [
        triad for index, triad in enumerate(all_triads)
        if index % args.num_jobs == args.job_index
    ]
    if args.max_triads is not None:
        triads = triads[: args.max_triads]
    sessions = sorted({session for triad in triads for session in triad[2:]})
    print(
        f"common target animal={args.animal} target={args.target} "
        f"target_epoch={args.target_epoch} job={args.job_index}/{args.num_jobs} "
        f"triads={len(triads)} repeats={args.repeats}",
        flush=True,
    )
    cache = {
        session: load_session(
            session, args.target, EXCLUDE_TRIALS.get(session, ())
        )
        for session in sessions
    }
    animal_offset = 0 if args.animal == "TS" else 50_000_000
    rows = []
    for triad_position, (
        r1_index, r2_index, r1_session, r2_session, target_session
    ) in enumerate(triads):
        r1 = cache[r1_session]
        r2 = cache[r2_session]
        target = cache[target_session]
        pair_index = r2_index * len(sessions_r1) + r1_index
        for repeat in range(args.repeats):
            split_seed = (
                SEED + animal_offset + pair_index * 100_000 + repeat * 1000
                + (0 if args.target_epoch == "R1" else 20_000_000)
            )
            folds_r1 = trial_folds(r1["meta"], N_FOLDS, split_seed)
            folds_r2 = trial_folds(r2["meta"], N_FOLDS, split_seed + 1)
            folds_target = trial_folds(target["meta"], N_FOLDS, split_seed + 2)
            for fold in range(N_FOLDS):
                rng_r1 = np.random.default_rng(split_seed + fold * 10 + 3)
                rng_r2 = np.random.default_rng(split_seed + fold * 10 + 4)
                rng_target = np.random.default_rng(split_seed + fold * 10 + 5)
                calibration_r1 = select_trials(
                    r1["meta"], ~folds_r1[fold],
                    N_SOURCE_CALIBRATION_TRIALS, rng_r1,
                )
                calibration_r2 = select_trials(
                    r2["meta"], ~folds_r2[fold],
                    N_SOURCE_CALIBRATION_TRIALS, rng_r2,
                )
                calibration_target = select_trials(
                    target["meta"], ~folds_target[fold],
                    N_TARGET_CALIBRATION_TRIALS, rng_target,
                )
                evaluation_target = select_trials(
                    target["meta"], folds_target[fold],
                    N_TARGET_EVALUATION_TRIALS, rng_target,
                )
                scores_r1 = score_source_to_target(
                    r1, target, calibration_r1, calibration_target,
                    evaluation_target, split_seed + fold * 2,
                )
                scores_r2 = score_source_to_target(
                    r2, target, calibration_r2, calibration_target,
                    evaluation_target, split_seed + fold * 2 + 1,
                )
                for decoder, variant in CONDITIONS:
                    rows.append({
                        "animal": args.animal,
                        "target": args.target,
                        "target_epoch": args.target_epoch,
                        "r1_source": r1_session,
                        "r2_source": r2_session,
                        "common_target": target_session,
                        "repeat": repeat,
                        "fold": fold,
                        "decoder": decoder,
                        "variant": variant,
                        "consensus_included": (
                            (decoder, variant) in PRIMARY_CONDITIONS
                        ),
                        "r1_source_score": scores_r1[(decoder, variant)],
                        "r2_source_score": scores_r2[(decoder, variant)],
                        "source_advantage": (
                            scores_r2[(decoder, variant)]
                            - scores_r1[(decoder, variant)]
                        ),
                    })
        print(
            f"common target {args.animal} {args.target} {args.target_epoch} "
            f"job {args.job_index}: {triad_position + 1}/{len(triads)} triads",
            flush=True,
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.max_triads is not None or args.repeats != REPEATS:
        suffix = "_smoke"
    output = OUT_DIR / (
        f"common_target_{args.animal.lower()}_{args.target}_"
        f"{args.target_epoch.lower()}_job_{args.job_index}_of_{args.num_jobs}"
        f"{suffix}.csv"
    )
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
