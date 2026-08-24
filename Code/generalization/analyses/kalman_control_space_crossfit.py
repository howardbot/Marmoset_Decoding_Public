"""Cross-fitted ridge and steady-state Kalman control-space geometry.

The Sadtler-inspired Kalman mapping is the direct neural term in

    x_hat[t] = (I - K_inf H) A x_hat[t-1] + K_inf z[t].

Because this analysis operates directly on canonical neural activity ``z``,
``M2 = K_inf``. Its row space is the Kalman-potent/control space. This is an
offline estimated mapping, not the externally fixed causal BCI mapping in
Sadtler et al. (2014).

Two analyses share the same trial-grouped 5-fold cross-fit:

* ``within`` fits each session independently and measures held-out neural
  variance in the ridge and Kalman potent spaces plus their angles to the top
  three calibration-PC axes.
* ``crossday`` fits calibration-only PCA/CCA for each session pair and measures
  principal angles between the two days' ridge or Kalman potent spaces.

The original concatenated-transition Kalman convention is used intentionally
to match the headline decoder. Evaluation kinematics never fit PCA, CCA,
readouts, or control spaces.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, str(THIS_DIR.parents[1]))

from big_sweep_phase2_crossday import EXCLUDE_TRIALS, SESSIONS_R1, SESSIONS_R2
from kalman_components import fit_kalman_components, steady_state_gain
from nested_cca import fit_pca_projector
from private_readout_crossfit import (
    K,
    L2,
    N_FOLDS,
    SEED,
    TARGET,
    fit_calibration_alignment,
    load_session,
    standardize_from_calibration,
    trial_folds,
)
from readout_subspaces import orthonormal_basis, readout_basis

REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "kalman_control_space"
REPEATS = 5
TOP_PC_RANK = 3


def ridge_fit(activity: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Return neural-by-state ridge weights for ``state ~ activity``.

    This is the ridge-defined output-potent/read-out map used throughout this
    file. If ``Z`` is canonical neural activity and ``X`` is the centered
    behavioral state, this solves

        W = (Z.T @ Z + lambda I)^-1 Z.T @ X

    so held-out state predictions would be ``X_hat = Z @ W``.

    ``W`` has shape ``n_neural_dims x n_state_dims``. Its column space is the
    ridge potent space in neural coordinates: activity inside this subspace can
    change the fixed linear read-out, while activity in the orthogonal
    complement is output-null for this read-out. The caller passes only
    calibration rows; evaluation trials are never used to fit ``W``.
    """
    return np.linalg.solve(
        activity.T @ activity + L2 * np.eye(activity.shape[1]),
        activity.T @ state,
    )


def principal_angles_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Principal angles between two subspaces, reported in degrees.

    The inputs may be arbitrary bases; they are orthonormalized first. Small
    angles mean the subspaces use similar neural directions. Large angles mean
    the neural directions used by one read-out/control space are rotated away
    from the other.
    """
    first = orthonormal_basis(first)
    second = orthonormal_basis(second)
    if first.shape[0] != second.shape[0]:
        raise ValueError("bases must use the same ambient space")
    cosines = np.linalg.svd(first.T @ second, compute_uv=False)
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def variance_fraction(activity: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of centered activity energy lying in an orthonormal basis."""
    activity = np.asarray(activity, dtype=float)
    basis = orthonormal_basis(basis)
    centered = activity - activity.mean(axis=0, keepdims=True)
    denominator = float(np.square(centered).sum())
    if denominator <= 0:
        return np.nan
    return float(np.square(centered @ basis).sum() / denominator)


def control_bases(
    activity: np.ndarray,
    state: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit ridge and original-Kalman potent bases in one neural coordinate system.

    Both inputs should already be restricted to calibration trials and should
    live in the same canonical neural coordinate system. The returned values
    are orthonormal neural-coordinate bases, so downstream code can project
    neural activity onto them or compare them with principal angles.
    """
    # Ridge potent space:
    #
    # ``ridge_weights`` maps canonical neural activity to the behavioral state.
    # Because it is ``neural_dim x state_dim``, ``readout_basis`` returns an
    # orthonormal basis for its neural column space. This is the usual
    # Kaufman-style output-potent space for a fitted linear read-out.
    ridge_weights = ridge_fit(activity, state)

    # Kalman potent/control space:
    #
    # ``fit_kalman_components`` estimates A/W/H/Q from the same calibration
    # data. ``steady_state_gain`` then iterates the covariance recursion to the
    # steady-state Kalman gain K_inf. In the direct canonical-activity update
    # used here,
    #
    #     x_hat[t] = (I - K_inf H) A x_hat[t-1] + K_inf z[t]
    #
    # the direct neural term is K_inf z[t]. Therefore row(K_inf) is the
    # Sadtler-inspired Kalman control/potent space. ``gain`` has shape
    # ``state_dim x neural_dim``; passing ``gain.T`` gives ``readout_basis`` a
    # ``neural_dim x state_dim`` matrix, whose column space equals row(K_inf).
    model = fit_kalman_components(activity, state, meta=None)
    gain, iterations, change = steady_state_gain(model)
    bases = {
        "ridge": readout_basis(ridge_weights),
        "kalman": readout_basis(gain.T),
    }
    diagnostics = {
        "kalman_iterations": iterations,
        "kalman_final_change": change,
        "kalman_gain_norm": float(np.linalg.norm(gain)),
    }
    return bases, diagnostics


def summarize_angles(prefix: str, angles: np.ndarray) -> dict[str, float]:
    """Store a compact summary of a vector of principal angles."""
    return {
        f"{prefix}_mean_angle_deg": float(np.mean(angles)),
        f"{prefix}_min_angle_deg": float(np.min(angles)),
        f"{prefix}_max_angle_deg": float(np.max(angles)),
        f"{prefix}_mean_cosine": float(np.mean(np.cos(np.radians(angles)))),
    }


def evaluate_within_split(data, calibration, evaluation) -> dict[str, float]:
    """Evaluate potent-space geometry for one session and one held-out fold.

    ``calibration`` and ``evaluation`` are boolean masks over trial-grouped
    samples. Calibration samples fit PCA, state centering/scaling, the ridge
    read-out, and the Kalman control space. Evaluation samples are used only
    for variance measurements, keeping the potent-fraction result non-circular.
    """
    # Fit PCA using calibration neural activity only, then transform all
    # samples so the held-out fold can be scored in the same PC coordinates.
    projector = fit_pca_projector(data["Y"][calibration], K)
    pc_activity = projector.transform(data["Y"])

    # Standardize PC activity with calibration statistics. The potent bases are
    # fitted in this standardized coordinate system so all PC dimensions have
    # comparable scale during fitting.
    calibration_mean = pc_activity[calibration].mean(axis=0)
    calibration_scale = pc_activity[calibration].std(axis=0)
    calibration_scale = np.where(calibration_scale > 1e-8, calibration_scale, 1.0)
    standardized = (pc_activity - calibration_mean) / calibration_scale

    # Center state by the calibration mean only. This avoids letting held-out
    # kinematics leak into the fitted read-out/control-space definitions.
    state = data["X"] - data["X"][calibration].mean(axis=0, keepdims=True)
    bases_standardized, diagnostics = control_bases(
        standardized[calibration], state[calibration]
    )

    row = dict(diagnostics)

    # The top-PC comparison asks whether the largest neural-variance axes are
    # close to the output-potent/control directions. Smaller angles indicate
    # stronger alignment between dominant neural variance and read-out use.
    top_pc = np.eye(K)[:, :TOP_PC_RANK]
    raw_bases = {}
    for method, basis_standardized in bases_standardized.items():
        # z = D^-1 y, so a direct mapping M z becomes M D^-1 y.
        # Convert its neural row space back to raw calibration-PC coordinates.
        raw_basis = orthonormal_basis(
            basis_standardized / calibration_scale[:, None]
        )
        raw_bases[method] = raw_basis

        # Held-out potent fraction: how much centered evaluation neural energy
        # lies inside this fitted potent/control subspace. This is a geometry
        # measurement of neural activity, not a decoder score.
        row[f"{method}_rank"] = raw_basis.shape[1]
        row[f"{method}_potent_fraction"] = variance_fraction(
            pc_activity[evaluation], raw_basis
        )
        row.update(summarize_angles(
            f"{method}_top_pc",
            principal_angles_deg(top_pc, raw_basis),
        ))

    # Paired diagnostic: if ridge and Kalman bases are nearly identical, the
    # Kalman control-space definition closes a method gap but does not provide
    # independent geometry beyond the ridge potent-space definition.
    row.update(summarize_angles(
        "ridge_kalman",
        principal_angles_deg(raw_bases["ridge"], raw_bases["kalman"]),
    ))
    return row


def evaluate_crossday_split(
    first,
    second,
    calibration_first,
    calibration_second,
    evaluation_first,
    evaluation_second,
    fit_seed: int,
) -> dict[str, float]:
    """Evaluate cross-day ridge/Kalman potent-space geometry for one split.

    This function fits the cross-day neural alignment using only calibration
    trials from each session, estimates each day's potent/control spaces in the
    shared canonical coordinate system, and scores all geometry on held-out
    trials. It asks whether R1-R2 potent spaces are unusually rotated relative
    to ordinary R1-R1 or R2-R2 day pairs.
    """
    # Calibration-only PCA/CCA alignment puts both sessions into a common
    # canonical neural space before fitting day-specific read-out spaces.
    alignment = fit_calibration_alignment(
        first,
        second,
        calibration_first,
        calibration_second,
        fit_seed,
    )

    # Each day is standardized by its own calibration distribution. This keeps
    # read-out fitting scale-controlled without using held-out samples.
    activity_first = standardize_from_calibration(
        alignment.transform_train(first["Y"]), calibration_first
    )
    activity_second = standardize_from_calibration(
        alignment.transform_target(second["Y"]), calibration_second
    )

    # State centering is also calibration-only and day-specific.
    state_first = first["X"] - first["X"][calibration_first].mean(
        axis=0, keepdims=True
    )
    state_second = second["X"] - second["X"][calibration_second].mean(
        axis=0, keepdims=True
    )

    # Fit ridge and Kalman potent/control bases separately for each day, but in
    # the same aligned neural coordinate system.
    bases_first, diagnostics_first = control_bases(
        activity_first[calibration_first], state_first[calibration_first]
    )
    bases_second, diagnostics_second = control_bases(
        activity_second[calibration_second], state_second[calibration_second]
    )

    row = {
        "kalman_iterations_first": diagnostics_first["kalman_iterations"],
        "kalman_iterations_second": diagnostics_second["kalman_iterations"],
        "kalman_final_change_first": diagnostics_first["kalman_final_change"],
        "kalman_final_change_second": diagnostics_second["kalman_final_change"],
        "kalman_gain_norm_first": diagnostics_first["kalman_gain_norm"],
        "kalman_gain_norm_second": diagnostics_second["kalman_gain_norm"],
    }
    for method in ("ridge", "kalman"):
        basis_first = bases_first[method]
        basis_second = bases_second[method]
        row[f"{method}_rank_first"] = basis_first.shape[1]
        row[f"{method}_rank_second"] = basis_second.shape[1]

        # Cross-day principal angles directly quantify potent-space rotation.
        # If the R1-R2 category were special, these angles should separate from
        # the R1-R1/R2-R2 baselines.
        row.update(summarize_angles(
            f"{method}_crossday",
            principal_angles_deg(basis_first, basis_second),
        ))

        # Own fractions are held-out neural variance in each day's own fitted
        # potent space. Cross fractions ask how much held-out activity from one
        # day lies in the other day's potent space after CCA alignment.
        row[f"{method}_first_own_fraction"] = variance_fraction(
            activity_first[evaluation_first], basis_first
        )
        row[f"{method}_second_own_fraction"] = variance_fraction(
            activity_second[evaluation_second], basis_second
        )
        row[f"{method}_first_in_second_fraction"] = variance_fraction(
            activity_first[evaluation_first], basis_second
        )
        row[f"{method}_second_in_first_fraction"] = variance_fraction(
            activity_second[evaluation_second], basis_first
        )

    # Within-day ridge-vs-Kalman comparisons verify whether the two definitions
    # actually identify different neural subspaces in this fitted model.
    row.update(summarize_angles(
        "first_ridge_kalman",
        principal_angles_deg(bases_first["ridge"], bases_first["kalman"]),
    ))
    row.update(summarize_angles(
        "second_ridge_kalman",
        principal_angles_deg(bases_second["ridge"], bases_second["kalman"]),
    ))
    return row


def session_specs() -> list[tuple[str, str]]:
    """Return labeled within-session jobs for all R1 and R2 days."""
    return [("R1", session) for session in SESSIONS_R1] + [
        ("R2", session) for session in SESSIONS_R2
    ]


def pair_specs() -> list[tuple[str, str, str]]:
    """Return labeled cross-day pairs used for category-level summaries."""
    return (
        [("R1-R1", first, second) for first, second in combinations(SESSIONS_R1, 2)]
        + [("R1-R2", first, second) for first, second in product(SESSIONS_R1, SESSIONS_R2)]
        + [("R2-R2", first, second) for first, second in combinations(SESSIONS_R2, 2)]
    )


def run_within(job_index: int, num_jobs: int, repeats: int, max_items: int | None):
    """Shardable within-session runner.

    Each session is split into repeated trial-grouped folds. Four folds define
    the potent/control spaces; the remaining fold scores held-out geometry.
    """
    specs = [
        item for index, item in enumerate(session_specs())
        if index % num_jobs == job_index
    ][:max_items]
    rows = []
    for item_index, (epoch, session) in enumerate(specs):
        data = load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        global_index = session_specs().index((epoch, session))
        for repeat in range(repeats):
            split_seed = SEED + global_index * 100_000 + repeat * 1000
            folds = trial_folds(data["meta"], N_FOLDS, split_seed)
            for fold, evaluation in enumerate(folds):
                row = evaluate_within_split(data, ~evaluation, evaluation)
                row.update({
                    "target": TARGET,
                    "epoch": epoch,
                    "session": session,
                    "repeat": repeat,
                    "fold": fold,
                })
                rows.append(row)
        print(
            f"within job {job_index + 1}/{num_jobs}: "
            f"completed {item_index + 1}/{len(specs)} sessions",
            flush=True,
        )
    return rows


def run_crossday(job_index: int, num_jobs: int, repeats: int, max_items: int | None):
    """Shardable cross-day runner over R1-R1, R1-R2, and R2-R2 pairs."""
    all_specs = pair_specs()
    specs = [
        item for index, item in enumerate(all_specs)
        if index % num_jobs == job_index
    ][:max_items]
    required_sessions = sorted({session for _, first, second in specs for session in (first, second)})
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in required_sessions
    }
    rows = []
    for item_index, (category, first_session, second_session) in enumerate(specs):
        first, second = cache[first_session], cache[second_session]
        global_index = all_specs.index((category, first_session, second_session))
        for repeat in range(repeats):
            split_seed = SEED + global_index * 100_000 + repeat * 1000
            folds_first = trial_folds(first["meta"], N_FOLDS, split_seed)
            folds_second = trial_folds(second["meta"], N_FOLDS, split_seed + 1)
            for fold in range(N_FOLDS):
                evaluation_first = folds_first[fold]
                evaluation_second = folds_second[fold]
                row = evaluate_crossday_split(
                    first,
                    second,
                    ~evaluation_first,
                    ~evaluation_second,
                    evaluation_first,
                    evaluation_second,
                    split_seed + fold,
                )
                row.update({
                    "target": TARGET,
                    "category": category,
                    "first_session": first_session,
                    "second_session": second_session,
                    "repeat": repeat,
                    "fold": fold,
                })
                rows.append(row)
        print(
            f"crossday job {job_index + 1}/{num_jobs}: "
            f"completed {item_index + 1}/{len(specs)} pairs",
            flush=True,
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=("within", "crossday"), required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--num-jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.job_index < args.num_jobs:
        raise ValueError("job-index must be in [0, num-jobs)")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = run_within if args.analysis == "within" else run_crossday
    rows = runner(args.job_index, args.num_jobs, args.repeats, args.max_items)
    suffix = "_smoke" if args.max_items is not None or args.repeats != REPEATS else ""
    output = OUT_DIR / (
        f"{args.analysis}_job_{args.job_index}_of_{args.num_jobs}{suffix}.csv"
    )
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {output} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
