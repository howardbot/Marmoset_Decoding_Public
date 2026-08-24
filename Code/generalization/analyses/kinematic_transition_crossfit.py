"""Neural-free cross-fit of directional one-step kinematic transition loss."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2
from kalman_components import fit_state_dynamics, transition_indices
from private_readout_crossfit import N_FOLDS, SEED, TARGET, load_session, trial_folds

REPO = _THIS.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "kalman_component_swap"


def fit_transition(state, calibration, meta):
    calibration_meta = meta[calibration].reset_index(drop=True)
    return fit_state_dynamics(state[calibration], calibration_meta)[0]


def one_step_nmse(A, state, evaluation, meta):
    evaluation_state = state[evaluation]
    evaluation_meta = meta[evaluation].reset_index(drop=True)
    source, target = transition_indices(evaluation_meta, len(evaluation_state))
    truth = evaluation_state[target]
    prediction = evaluation_state[source] @ A.T
    denominator = np.var(truth, axis=0)
    valid = denominator > 1e-12
    return float(np.mean(np.mean((truth[:, valid] - prediction[:, valid]) ** 2, axis=0)
                         / denominator[valid]))


def evaluate_pair(a, b, r2_index, pair_index, repeats):
    rows = []
    for repeat in range(repeats):
        split_seed = SEED + r2_index * 1_000_000 + pair_index * 10_000 + repeat * 100
        folds_a = trial_folds(a["meta"], N_FOLDS, split_seed)
        folds_b = trial_folds(b["meta"], N_FOLDS, split_seed + 1)
        for fold in range(N_FOLDS):
            evaluation_a = folds_a[fold]
            evaluation_b = folds_b[fold]
            calibration_a = ~evaluation_a
            calibration_b = ~evaluation_b

            center_a = a["X"][calibration_a].mean(axis=0)
            forward_a = a["X"] - center_a
            forward_b = b["X"] - center_a
            center_b = b["X"][calibration_b].mean(axis=0)
            reverse_b = b["X"] - center_b
            reverse_a = a["X"] - center_b

            A_forward_source = fit_transition(
                forward_a, calibration_a, a["meta"]
            )
            A_forward_target = fit_transition(
                forward_b, calibration_b, b["meta"]
            )
            A_reverse_source = fit_transition(
                reverse_b, calibration_b, b["meta"]
            )
            A_reverse_target = fit_transition(
                reverse_a, calibration_a, a["meta"]
            )
            fwd_source = one_step_nmse(
                A_forward_source, forward_b, evaluation_b, b["meta"]
            )
            fwd_adapted = one_step_nmse(
                A_forward_target, forward_b, evaluation_b, b["meta"]
            )
            rev_source = one_step_nmse(
                A_reverse_source, reverse_a, evaluation_a, a["meta"]
            )
            rev_adapted = one_step_nmse(
                A_reverse_target, reverse_a, evaluation_a, a["meta"]
            )
            rows.append({
                "repeat": repeat,
                "fold": fold,
                "fwd_source_nmse": fwd_source,
                "fwd_adapted_nmse": fwd_adapted,
                "rev_source_nmse": rev_source,
                "rev_adapted_nmse": rev_adapted,
                "fwd_transition_penalty": fwd_source - fwd_adapted,
                "rev_transition_penalty": rev_source - rev_adapted,
                "directional_transition_penalty": (
                    (fwd_source - fwd_adapted) - (rev_source - rev_adapted)
                ),
            })
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
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
    sessions = sorted({
        session
        for _, _, r1_session, r2_session in pairs
        for session in (r1_session, r2_session)
    })
    print(
        f"loading kinematic job {args.job_index + 1}/{args.num_jobs}: "
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
        pair_rows = evaluate_pair(
            cache[r1_session], cache[r2_session], r2_index, pair_index, args.repeats
        )
        for row in pair_rows:
            row.update({"r1_session": r1_session, "r2_session": r2_session})
            rows.append(row)
        print(
            f"kinematic job {args.job_index + 1}/{args.num_jobs}: "
            f"completed {completed}/{len(pairs)}",
            flush=True,
        )
    output = OUT_DIR / f"kinematic_transition_job_{args.job_index}_of_{args.num_jobs}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
