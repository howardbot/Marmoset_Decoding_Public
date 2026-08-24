"""Test whether original-Kalman asymmetry depends on arbitrary trial order.

The original Kording fit treats the calibration array as one continuous
sequence, so every adjacent pair of trial blocks contributes one false state
transition.  This falsifier preserves every within-trial sample and its order,
but randomly permutes whole calibration-trial blocks before fitting A/W/H/Q.
Prediction remains trial-by-trial, exactly as in the headline pipeline.

Random ordering tests sensitivity to *which* false boundaries are present.  It
does not remove all false boundaries; a stable result therefore rules out
order-specific contamination, not boundary contamination as a class.
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

from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2
from decoder_model_audit import score_kalman
from kalman_component_swap import source_centered_states
from kalman_components import fit_kalman_components
from private_readout_crossfit import (
    N_FOLDS,
    SEED,
    fit_calibration_alignment,
    load_session,
    standardize_from_calibration,
    trial_folds,
)

REPO = _THIS.parents[2]
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "kalman_trial_order"
TARGET = "relative_position"
REPEATS = 5
N_PERMUTATIONS = 20


def permuted_trial_indices(meta: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Return row indices after shuffling intact trial blocks."""
    blocks = [
        np.asarray(indices, dtype=int)
        for indices in meta.groupby("trial_number", sort=False).indices.values()
    ]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order])


def relative_matrix_change(permuted: np.ndarray, original: np.ndarray) -> float:
    return float(
        np.linalg.norm(permuted - original, ord="fro")
        / (np.linalg.norm(original, ord="fro") + 1e-12)
    )


def evaluate_split(
    a,
    b,
    calibration_a,
    calibration_b,
    evaluation_a,
    evaluation_b,
    fit_seed: int,
    n_permutations: int,
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

    calibration_meta_a = a["meta"][calibration_a].reset_index(drop=True)
    calibration_meta_b = b["meta"][calibration_b].reset_index(drop=True)
    calibration_activity_a = activity_a[calibration_a]
    calibration_activity_b = activity_b[calibration_b]
    calibration_state_a = forward_state_a[calibration_a]
    calibration_state_b = reverse_state_b[calibration_b]

    baseline_forward = fit_kalman_components(
        calibration_activity_a, calibration_state_a
    )
    baseline_reverse = fit_kalman_components(
        calibration_activity_b, calibration_state_b
    )
    rows = []

    def score_models(forward_model, reverse_model, permutation: int):
        fwd_native, fwd_common = score_kalman(
            forward_model,
            activity_b,
            forward_state_b,
            evaluation_b,
            b["meta"],
        )
        rev_native, rev_common = score_kalman(
            reverse_model,
            activity_a,
            reverse_state_a,
            evaluation_a,
            a["meta"],
        )
        return {
            "permutation": permutation,
            "fwd_native": fwd_native,
            "rev_native": rev_native,
            "gap_native": rev_native - fwd_native,
            "fwd_common": fwd_common,
            "rev_common": rev_common,
            "gap_common": rev_common - fwd_common,
            "fwd_A_relative_change": relative_matrix_change(
                forward_model.A, baseline_forward.A
            ),
            "fwd_W_relative_change": relative_matrix_change(
                forward_model.W, baseline_forward.W
            ),
            "rev_A_relative_change": relative_matrix_change(
                reverse_model.A, baseline_reverse.A
            ),
            "rev_W_relative_change": relative_matrix_change(
                reverse_model.W, baseline_reverse.W
            ),
        }

    rows.append(score_models(baseline_forward, baseline_reverse, -1))
    for permutation in range(n_permutations):
        forward_order = permuted_trial_indices(
            calibration_meta_a,
            np.random.default_rng(fit_seed + 100_000 + permutation * 2),
        )
        reverse_order = permuted_trial_indices(
            calibration_meta_b,
            np.random.default_rng(fit_seed + 100_001 + permutation * 2),
        )
        forward_model = fit_kalman_components(
            calibration_activity_a[forward_order],
            calibration_state_a[forward_order],
        )
        reverse_model = fit_kalman_components(
            calibration_activity_b[reverse_order],
            calibration_state_b[reverse_order],
        )
        rows.append(score_models(forward_model, reverse_model, permutation))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r2-index", required=True, type=int, choices=range(len(SESSIONS_R2))
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r2_session = SESSIONS_R2[args.r2_index]
    r1_sessions = SESSIONS_R1[: args.max_pairs]
    sessions = list(r1_sessions) + [r2_session]
    print(
        f"trial-order falsifier R2[{args.r2_index}] pairs={len(r1_sessions)} "
        f"repeats={args.repeats} permutations={args.permutations}",
        flush=True,
    )
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
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
                    args.permutations,
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
            f"trial-order R2[{args.r2_index}]: "
            f"{pair_index + 1}/{len(r1_sessions)} pairs",
            flush=True,
        )

    suffix = (
        "_smoke"
        if args.max_pairs is not None
        or args.repeats != REPEATS
        or args.permutations != N_PERMUTATIONS
        else ""
    )
    output = OUT_DIR / f"trial_order_r2_{args.r2_index}{suffix}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
