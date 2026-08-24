"""Final cross-fitted decomposition of the Kalman neural observation map H."""
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

from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2
from h_observation_decomposition import (
    ObservationModel,
    apply_column_mask,
    fit_observation_model,
    split_observation_delta,
)
from kalman_component_swap import (
    fit_day_components,
    score_model,
    source_centered_states,
)
from kalman_components import KalmanComponents
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
    principal_readout_subspaces,
    random_subspace_within,
    readout_basis,
)


REPO = _THIS.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "h_observation_fine_swap"
OBSERVATION_FAMILIES = ("no_intercept", "affine")
Q_CONTEXTS = ("source", "target")
AXIS_NAMES = ("x", "y", "z")
THRESHOLD = 0.5
DEFAULT_RANDOM_DRAWS = 3


def observation_families(components, activity, state, calibration):
    no_intercept = ObservationModel(
        H=components.H,
        Q=components.Q,
        b=np.zeros(components.H.shape[0]),
    )
    affine = fit_observation_model(
        activity[calibration], state[calibration], affine=True
    )
    return {"no_intercept": no_intercept, "affine": affine}


def score_observation(
    source_components,
    H,
    Q,
    b,
    activity,
    state,
    evaluation,
    meta,
):
    model = KalmanComponents(
        A=source_components.A,
        W=source_components.W,
        H=np.asarray(H),
        Q=np.asarray(Q),
    )
    return score_model(
        model,
        np.asarray(activity) - np.asarray(b)[None, :],
        state,
        evaluation,
        meta,
    )


def delta_fraction(delta, component):
    denominator = float(np.sum(np.asarray(delta) ** 2))
    if denominator < 1e-20:
        return np.nan
    return float(np.sum(np.asarray(component) ** 2) / denominator)


def append_direction_rows(
    stores,
    direction,
    source_components,
    source_observations,
    target_observations,
    activity,
    state,
    evaluation,
    meta,
    private_basis,
    random_bases,
):
    for family in OBSERVATION_FAMILIES:
        source = source_observations[family]
        target = target_observations[family]
        delta = target.H - source.H
        private_delta, rest_delta = split_observation_delta(
            source.H, target.H, private_basis
        )
        reconstruction_error = float(np.max(np.abs(
            source.H + private_delta + rest_delta - target.H
        )))
        if reconstruction_error > 1e-10:
            raise AssertionError(
                f"private/rest H reconstruction failed: {reconstruction_error}"
            )
        random_deltas = [
            split_observation_delta(source.H, target.H, basis)[0]
            for basis in random_bases
        ]
        partition_conditions = [
            ("source", source.H, -1, 0.0),
            (
                "private_delta",
                source.H + private_delta,
                -1,
                delta_fraction(delta, private_delta),
            ),
            (
                "rest_delta",
                source.H + rest_delta,
                -1,
                delta_fraction(delta, rest_delta),
            ),
            ("target_H", target.H, -1, 1.0),
        ]
        partition_conditions.extend([
            (
                "random_delta",
                source.H + random_delta,
                draw,
                delta_fraction(delta, random_delta),
            )
            for draw, random_delta in enumerate(random_deltas)
        ])
        for q_context in Q_CONTEXTS:
            Q = source.Q if q_context == "source" else target.Q
            for condition, H, random_draw, fraction in partition_conditions:
                stores["partition"].append({
                    "direction": direction,
                    "observation_family": family,
                    "q_context": q_context,
                    "condition": condition,
                    "random_draw": random_draw,
                    "rank_private": private_basis.shape[1],
                    "delta_fraction": fraction,
                    "score": score_observation(
                        source_components,
                        H,
                        Q,
                        source.b,
                        activity,
                        state,
                        evaluation,
                        meta,
                    ),
                })

        if family == "no_intercept":
            for q_context in Q_CONTEXTS:
                Q = source.Q if q_context == "source" else target.Q
                for delta_scope, scoped_delta in (
                    ("full", delta),
                    ("private", private_delta),
                ):
                    for axis_mask in range(2 ** len(AXIS_NAMES)):
                        H = apply_column_mask(source.H, scoped_delta, axis_mask)
                        endpoint = source.H + scoped_delta
                        if axis_mask == 2 ** len(AXIS_NAMES) - 1:
                            error = float(np.max(np.abs(H - endpoint)))
                            if error > 1e-10:
                                raise AssertionError(
                                    f"axis H reconstruction failed: {error}"
                                )
                        stores["axis"].append({
                            "direction": direction,
                            "observation_family": family,
                            "q_context": q_context,
                            "delta_scope": delta_scope,
                            "axis_mask": axis_mask,
                            "target_axes": "+".join([
                                name for index, name in enumerate(AXIS_NAMES)
                                if axis_mask & (1 << index)
                            ]) or "source",
                            "score": score_observation(
                                source_components,
                                H,
                                Q,
                                source.b,
                                activity,
                                state,
                                evaluation,
                                meta,
                            ),
                        })

        if family == "affine":
            for q_context in Q_CONTEXTS:
                Q = source.Q if q_context == "source" else target.Q
                for affine_mask in range(4):
                    b = target.b if affine_mask & 1 else source.b
                    H = target.H if affine_mask & 2 else source.H
                    stores["affine"].append({
                        "direction": direction,
                        "observation_family": family,
                        "q_context": q_context,
                        "affine_mask": affine_mask,
                        "target_parts": "+".join([
                            name for index, name in enumerate(("b", "H"))
                            if affine_mask & (1 << index)
                        ]) or "source",
                        "score": score_observation(
                            source_components,
                            H,
                            Q,
                            b,
                            activity,
                            state,
                            evaluation,
                            meta,
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
    random_draws,
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
    forward_target = fit_day_components(
        activity_b, forward_state_b, calibration_b, b["meta"], "trial_aware"
    )
    reverse_source = fit_day_components(
        activity_b, reverse_state_b, calibration_b, b["meta"], "trial_aware"
    )
    reverse_target = fit_day_components(
        activity_a, reverse_state_a, calibration_a, a["meta"], "trial_aware"
    )
    forward_source_observations = observation_families(
        forward_source, activity_a, forward_state_a, calibration_a
    )
    forward_target_observations = observation_families(
        forward_target, activity_b, forward_state_b, calibration_b
    )
    reverse_source_observations = observation_families(
        reverse_source, activity_b, reverse_state_b, calibration_b
    )
    reverse_target_observations = observation_families(
        reverse_target, activity_a, reverse_state_a, calibration_a
    )

    basis_a = readout_basis(
        ridge_fit(activity_a[calibration_a], a["X"][calibration_a])
    )
    basis_b = readout_basis(
        ridge_fit(activity_b[calibration_b], b["X"][calibration_b])
    )
    spaces = principal_readout_subspaces(basis_a, basis_b, THRESHOLD)
    random_bases = []
    for draw in range(random_draws):
        rng = np.random.default_rng(fit_seed + draw * 1000 + 9101)
        random_bases.append(random_subspace_within(
            spaces.union, spaces.private_a.shape[1], rng
        ))

    stores = {"partition": [], "axis": [], "affine": []}
    append_direction_rows(
        stores,
        "forward",
        forward_source,
        forward_source_observations,
        forward_target_observations,
        activity_b,
        forward_state_b,
        evaluation_b,
        b["meta"],
        spaces.private_a,
        random_bases,
    )
    append_direction_rows(
        stores,
        "reverse",
        reverse_source,
        reverse_source_observations,
        reverse_target_observations,
        activity_a,
        reverse_state_a,
        evaluation_a,
        a["meta"],
        spaces.private_a,
        random_bases,
    )
    return stores


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
        f"loading H fine-swap job {args.job_index + 1}/{args.num_jobs}: "
        f"{len(pairs)} pairs",
        flush=True,
    )
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    stores = {"partition": [], "axis": [], "affine": []}
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
                split_stores = evaluate_split(
                    a,
                    b,
                    ~evaluation_a,
                    ~evaluation_b,
                    evaluation_a,
                    evaluation_b,
                    split_seed + fold,
                    args.random_draws,
                )
                identifiers = {
                    "target": TARGET,
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "repeat": repeat,
                    "fold": fold,
                }
                for name, rows in split_stores.items():
                    for row in rows:
                        row.update(identifiers)
                        stores[name].append(row)
        print(
            f"H fine-swap job {args.job_index + 1}/{args.num_jobs}: "
            f"completed {completed}/{len(pairs)}",
            flush=True,
        )
    suffix = f"job_{args.job_index}_of_{args.num_jobs}"
    if args.max_pairs is not None:
        suffix += "_smoke"
    for name, rows in stores.items():
        output = OUT_DIR / f"h_{name}_{suffix}.csv"
        pd.DataFrame(rows).to_csv(output, index=False)
        print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
