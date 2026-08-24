"""Cross-fitted trial-removal dose response for residual neural variability.

This is the leakage-controlled follow-up to ``remove_to_match_neural_variance``.
For every R1/R2 session pair and outer trial fold, calibration trials alone fit
the session PCAs, phase-matched CCA, kinematics-to-neural residual models,
residual covariance transport, and a neural-only denoiser.  The denoiser gives
each held-out trial a removal score without reading held-out kinematics.

Calibration and evaluation trials are Hungarian-matched separately by their
position + velocity trajectories.  The kinematic feature scaler is fitted on
calibration trials.  A cutoff learned on calibration pairs at each prespecified
retention fraction is transferred unchanged to evaluation pairs; evaluation
data are never searched for the subset that best matches variance or closes the
decoder gap.  Removing a pair always removes both kinematically paired trials.

For each dose, equal-count random paired removal is compared with three
factorial interventions:

``variance_train_only``
    Variance-selected calibration pairs and random evaluation pairs.
``variance_eval_only``
    Random calibration pairs and variance-selected evaluation pairs.
``variance_both``
    Variance-selected calibration and evaluation pairs.
``random_both``
    Equal-count random calibration and evaluation pairs.

The primary selector is ``neural_only``.  An optional ``oracle_residual``
selector uses held-out movement to score held-out residual energy and is clearly
marked as target-conditioned sensitivity analysis, not the primary result.

The full analysis is shardable by R2 day.  Run each ``--r2-index`` and then use
``--summarize-only`` to create pair-, R2-day-, and experiment-level summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

from big_sweep_phase2_crossday import (  # noqa: E402
    EXCLUDE_TRIALS,
    SESSIONS_R1,
    SESSIONS_R2,
    kalman_fit_predict,
    m2_per_trial,
)
from nested_cca_validation import load_session  # noqa: E402
from private_readout_crossfit import (  # noqa: E402
    fit_calibration_alignment,
    trial_folds,
)
from r2_residual_covariance_denoising import (  # noqa: E402
    covariance_transport,
    fit_kinematic_signal,
    fit_neural_only_denoiser,
    kinematic_design,
    regularized_covariance,
)

warnings.filterwarnings("ignore")

K = 12
TARGET = "relative_position"
N_PHASE_MATCH = 30
N_FOLDS = 5
N_REPEATS = 5
N_RANDOM_REPS = 10
SEED = 20260803
RETAIN_FRACTIONS = (1.0, 0.8, 0.6, 0.5, 0.4)
PRIMARY_SELECTOR = "neural_only"
VALID_SELECTORS = (PRIMARY_SELECTOR, "oracle_residual")
MIN_CALIBRATION_PAIRS = 10
MIN_EVALUATION_PAIRS = 3

REPO = _THIS.parents[2]
DEFAULT_OUT_DIR = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "variance_removal_dose_response_crossfit"
)


@dataclass(frozen=True)
class MatchScaler:
    mean: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class TrialPairs:
    r1_trials: np.ndarray
    r2_trials: np.ndarray
    distance: np.ndarray

    def __len__(self) -> int:
        return len(self.r1_trials)

    def subset(self, indices: np.ndarray) -> "TrialPairs":
        indices = np.asarray(indices, dtype=int)
        return TrialPairs(
            r1_trials=self.r1_trials[indices],
            r2_trials=self.r2_trials[indices],
            distance=self.distance[indices],
        )


def short_session(session: str) -> str:
    match = re.search(r"2025(\d{4})", session)
    return match.group(1) if match else session


def stable_hash(*values: np.ndarray | str) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, str):
            digest.update(value.encode("utf-8"))
        else:
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.view(np.uint8))
    return digest.hexdigest()[:16]


def fisher_z(correlation: float) -> float:
    if not np.isfinite(correlation):
        return np.nan
    return float(np.arctanh(np.clip(correlation, -0.999999, 0.999999)))


def exact_sign_flip_p(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = abs(values.mean())
    signs = np.asarray(list(product([-1.0, 1.0], repeat=len(values))))
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= observed - 1e-12))


def mask_for_trials(meta: pd.DataFrame, trials) -> np.ndarray:
    return meta["trial_number"].isin(list(np.asarray(trials))).to_numpy()


def _trial_ids(meta: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return np.asarray(sorted(meta.loc[mask, "trial_number"].unique()))


def trial_trajectory_map(
    values: np.ndarray,
    meta: pd.DataFrame,
    trial_ids,
    n_phase: int = N_PHASE_MATCH,
) -> dict:
    """Phase-resample complete trials without fitting any parameters."""
    values = np.asarray(values, dtype=float)
    all_trials = meta["trial_number"].to_numpy()
    target_phase = np.linspace(0.0, 1.0, n_phase)
    trajectories = {}
    for trial in np.asarray(trial_ids):
        indices = np.flatnonzero(all_trials == trial)
        if len(indices) < 3:
            continue
        source_phase = np.linspace(0.0, 1.0, len(indices))
        trajectories[trial] = np.column_stack([
            np.interp(target_phase, source_phase, values[indices, dimension])
            for dimension in range(values.shape[1])
        ])
    return trajectories


def _flatten_trajectories(trajectories: dict) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(sorted(trajectories), dtype=object)
    if len(ids) == 0:
        raise ValueError("No usable trials were available for phase matching")
    features = np.stack([trajectories[trial].ravel() for trial in ids])
    return ids, features


def fit_match_scaler(
    kinematics_r1: np.ndarray,
    meta_r1: pd.DataFrame,
    calibration_r1: np.ndarray,
    kinematics_r2: np.ndarray,
    meta_r2: pd.DataFrame,
    calibration_r2: np.ndarray,
) -> MatchScaler:
    trajectories_r1 = trial_trajectory_map(
        kinematics_r1, meta_r1, _trial_ids(meta_r1, calibration_r1)
    )
    trajectories_r2 = trial_trajectory_map(
        kinematics_r2, meta_r2, _trial_ids(meta_r2, calibration_r2)
    )
    _, features_r1 = _flatten_trajectories(trajectories_r1)
    _, features_r2 = _flatten_trajectories(trajectories_r2)
    pooled = np.vstack([features_r1, features_r2])
    mean = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return MatchScaler(mean=mean, scale=scale)


def match_trial_sets(
    kinematics_r1: np.ndarray,
    meta_r1: pd.DataFrame,
    mask_r1: np.ndarray,
    kinematics_r2: np.ndarray,
    meta_r2: pd.DataFrame,
    mask_r2: np.ndarray,
    scaler: MatchScaler,
) -> TrialPairs:
    """Hungarian-match one-to-one trial paths using a frozen scaler."""
    trajectories_r1 = trial_trajectory_map(
        kinematics_r1, meta_r1, _trial_ids(meta_r1, mask_r1)
    )
    trajectories_r2 = trial_trajectory_map(
        kinematics_r2, meta_r2, _trial_ids(meta_r2, mask_r2)
    )
    ids_r1, features_r1 = _flatten_trajectories(trajectories_r1)
    ids_r2, features_r2 = _flatten_trajectories(trajectories_r2)
    standardized_r1 = (features_r1 - scaler.mean) / scaler.scale
    standardized_r2 = (features_r2 - scaler.mean) / scaler.scale
    cost = cdist(standardized_r1, standardized_r2) / np.sqrt(
        standardized_r1.shape[1]
    )
    index_r1, index_r2 = linear_sum_assignment(cost)
    return TrialPairs(
        r1_trials=ids_r1[index_r1],
        r2_trials=ids_r2[index_r2],
        distance=cost[index_r1, index_r2],
    )


def trial_variance(stack: np.ndarray) -> float:
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3 or len(stack) < 2:
        return np.nan
    centered = stack - stack.mean(axis=0, keepdims=True)
    return float(np.mean(centered**2))


def paired_residual_variance(
    residual_trajectories_r1: dict,
    residual_trajectories_r2: dict,
    pairs: TrialPairs,
) -> tuple[float, float, float]:
    stack_r1 = np.stack([
        residual_trajectories_r1[trial] for trial in pairs.r1_trials
    ])
    stack_r2 = np.stack([
        residual_trajectories_r2[trial] for trial in pairs.r2_trials
    ])
    variance_r1 = trial_variance(stack_r1)
    variance_r2 = trial_variance(stack_r2)
    ratio = variance_r2 / variance_r1 if variance_r1 > 0 else np.nan
    return variance_r1, variance_r2, ratio


def trial_change_scores(
    activity: np.ndarray,
    quiet_activity: np.ndarray,
    meta: pd.DataFrame,
    trial_ids,
) -> dict:
    """Neural-only per-trial score; this signature intentionally has no kinematics."""
    activity = np.asarray(activity, dtype=float)
    quiet_activity = np.asarray(quiet_activity, dtype=float)
    if activity.shape != quiet_activity.shape:
        raise ValueError("activity and quiet_activity must share a shape")
    trial_column = meta["trial_number"].to_numpy()
    scores = {}
    for trial in np.asarray(trial_ids):
        indices = np.flatnonzero(trial_column == trial)
        if len(indices) == 0:
            raise ValueError(f"Trial {trial!r} is absent from metadata")
        scores[trial] = float(np.mean((activity[indices] - quiet_activity[indices]) ** 2))
    return scores


def trial_oracle_residual_scores(
    residual: np.ndarray,
    meta: pd.DataFrame,
    trial_ids,
) -> dict:
    """Target-conditioned residual-energy score used only as an oracle bound."""
    residual = np.asarray(residual, dtype=float)
    trial_column = meta["trial_number"].to_numpy()
    scores = {}
    for trial in np.asarray(trial_ids):
        indices = np.flatnonzero(trial_column == trial)
        if len(indices) == 0:
            raise ValueError(f"Trial {trial!r} is absent from metadata")
        scores[trial] = float(np.mean(residual[indices] ** 2))
    return scores


def scores_for_pairs(pairs: TrialPairs, scores: dict, high_side: str) -> np.ndarray:
    if high_side == "r1":
        trial_ids = pairs.r1_trials
    elif high_side == "r2":
        trial_ids = pairs.r2_trials
    else:
        raise ValueError("high_side must be 'r1' or 'r2'")
    return np.asarray([scores[trial] for trial in trial_ids], dtype=float)


def calibration_cutoff_selection(
    calibration_scores: np.ndarray,
    evaluation_scores: np.ndarray,
    retention_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Learn a numeric cutoff on calibration and transfer it to evaluation."""
    calibration_scores = np.asarray(calibration_scores, dtype=float)
    evaluation_scores = np.asarray(evaluation_scores, dtype=float)
    if not 0 < retention_fraction <= 1:
        raise ValueError("retention_fraction must be in (0, 1]")
    if len(calibration_scores) == 0 or len(evaluation_scores) == 0:
        raise ValueError("calibration and evaluation scores must be nonempty")
    if retention_fraction == 1.0:
        return (
            np.arange(len(calibration_scores)),
            np.arange(len(evaluation_scores)),
            np.inf,
        )
    count = max(1, int(np.ceil(retention_fraction * len(calibration_scores))))
    order = np.argsort(calibration_scores, kind="mergesort")
    calibration_keep = np.sort(order[:count])
    cutoff = float(calibration_scores[order[count - 1]])
    evaluation_keep = np.flatnonzero(evaluation_scores <= cutoff)
    return calibration_keep, evaluation_keep, cutoff


def random_pair_indices(n_pairs: int, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    if not 0 <= n_keep <= n_pairs:
        raise ValueError("n_keep must lie between zero and n_pairs")
    return np.sort(rng.choice(n_pairs, n_keep, replace=False))


def factorial_pair_sets(
    variance_calibration: TrialPairs,
    variance_evaluation: TrialPairs,
    random_calibration: TrialPairs,
    random_evaluation: TrialPairs,
) -> dict[str, tuple[TrialPairs, TrialPairs]]:
    """Pure routing helper for the paired 2 x 2 intervention."""
    return {
        "random_both": (random_calibration, random_evaluation),
        "variance_train_only": (variance_calibration, random_evaluation),
        "variance_eval_only": (random_calibration, variance_evaluation),
        "variance_both": (variance_calibration, variance_evaluation),
    }


def decode_pair_sets(
    r1: dict,
    r2: dict,
    activity_r1: np.ndarray,
    activity_r2: np.ndarray,
    calibration_pairs: TrialPairs,
    evaluation_pairs: TrialPairs,
) -> dict:
    calibration_r1 = mask_for_trials(r1["meta"], calibration_pairs.r1_trials)
    calibration_r2 = mask_for_trials(r2["meta"], calibration_pairs.r2_trials)
    evaluation_r1 = mask_for_trials(r1["meta"], evaluation_pairs.r1_trials)
    evaluation_r2 = mask_for_trials(r2["meta"], evaluation_pairs.r2_trials)
    meta_evaluation_r1 = r1["meta"].loc[evaluation_r1].reset_index(drop=True)
    meta_evaluation_r2 = r2["meta"].loc[evaluation_r2].reset_index(drop=True)

    state_forward, prediction_forward = kalman_fit_predict(
        r1["X"][calibration_r1],
        activity_r1[calibration_r1],
        r2["X"][evaluation_r2],
        activity_r2[evaluation_r2],
        meta_evaluation_r2,
    )
    state_reverse, prediction_reverse = kalman_fit_predict(
        r2["X"][calibration_r2],
        activity_r2[calibration_r2],
        r1["X"][evaluation_r1],
        activity_r1[evaluation_r1],
        meta_evaluation_r1,
    )
    state_own_r1, prediction_own_r1 = kalman_fit_predict(
        r1["X"][calibration_r1],
        activity_r1[calibration_r1],
        r1["X"][evaluation_r1],
        activity_r1[evaluation_r1],
        meta_evaluation_r1,
    )
    state_own_r2, prediction_own_r2 = kalman_fit_predict(
        r2["X"][calibration_r2],
        activity_r2[calibration_r2],
        r2["X"][evaluation_r2],
        activity_r2[evaluation_r2],
        meta_evaluation_r2,
    )
    forward = m2_per_trial(state_forward, prediction_forward, meta_evaluation_r2)
    reverse = m2_per_trial(state_reverse, prediction_reverse, meta_evaluation_r1)
    own_r1 = m2_per_trial(state_own_r1, prediction_own_r1, meta_evaluation_r1)
    own_r2 = m2_per_trial(state_own_r2, prediction_own_r2, meta_evaluation_r2)
    return {
        "forward_corr": forward,
        "reverse_corr": reverse,
        "gap_corr": reverse - forward,
        "gap_fisher_z": fisher_z(reverse) - fisher_z(forward),
        "own_r1_corr": own_r1,
        "own_r2_corr": own_r2,
    }


def _alignment_hash(alignment) -> str:
    return stable_hash(
        alignment.train_pca.mean,
        alignment.train_pca.components,
        alignment.target_pca.mean,
        alignment.target_pca.components,
        alignment.train_rotation,
        alignment.target_rotation,
        alignment.train_cca_mean,
        alignment.target_cca_mean,
    )


def fit_neural_only_gate(
    activity_r1: np.ndarray,
    activity_r2: np.ndarray,
    signal_r1: np.ndarray,
    signal_r2: np.ndarray,
    residual_r1: np.ndarray,
    residual_r2: np.ndarray,
    calibration_r1: np.ndarray,
    calibration_r2: np.ndarray,
    high_side: str,
) -> tuple[np.ndarray, dict]:
    """Fit the high-variance-to-low-variance denoiser on calibration only."""
    covariance_r1 = regularized_covariance(residual_r1[calibration_r1])
    covariance_r2 = regularized_covariance(residual_r2[calibration_r2])
    if high_side == "r2":
        transform, eigenvalues = covariance_transport(
            covariance_r2, covariance_r1, shrink_only=True
        )
        calibration_target = (
            signal_r2[calibration_r2] + residual_r2[calibration_r2] @ transform
        )
        quiet = fit_neural_only_denoiser(
            activity_r2, calibration_target, calibration_r2
        )
        source = activity_r2
    elif high_side == "r1":
        transform, eigenvalues = covariance_transport(
            covariance_r1, covariance_r2, shrink_only=True
        )
        calibration_target = (
            signal_r1[calibration_r1] + residual_r1[calibration_r1] @ transform
        )
        quiet = fit_neural_only_denoiser(
            activity_r1, calibration_target, calibration_r1
        )
        source = activity_r1
    else:
        raise ValueError("high_side must be 'r1' or 'r2'")
    diagnostics = {
        "gate_fit_id": stable_hash(high_side, transform, source[calibration_r1 if high_side == 'r1' else calibration_r2], quiet[calibration_r1 if high_side == 'r1' else calibration_r2]),
        "generalized_eigenvalue_min": float(np.min(eigenvalues)),
        "generalized_eigenvalue_max": float(np.max(eigenvalues)),
        "n_directions_shrunk": int(np.sum(eigenvalues < 1.0)),
    }
    return quiet, diagnostics


def _row_diagnostics(
    calibration_pairs: TrialPairs,
    evaluation_pairs: TrialPairs,
    residual_trajectories_r1: dict,
    residual_trajectories_r2: dict,
) -> dict:
    cal_v1, cal_v2, cal_ratio = paired_residual_variance(
        residual_trajectories_r1, residual_trajectories_r2, calibration_pairs
    )
    eval_v1, eval_v2, eval_ratio = paired_residual_variance(
        residual_trajectories_r1, residual_trajectories_r2, evaluation_pairs
    )
    return {
        "n_calibration_pairs_retained": len(calibration_pairs),
        "n_evaluation_pairs_retained": len(evaluation_pairs),
        "calibration_residual_variance_r1": cal_v1,
        "calibration_residual_variance_r2": cal_v2,
        "calibration_residual_variance_ratio": cal_ratio,
        "evaluation_residual_variance_r1": eval_v1,
        "evaluation_residual_variance_r2": eval_v2,
        "evaluation_residual_variance_ratio": eval_ratio,
        "calibration_kinematic_distance": float(np.mean(calibration_pairs.distance)),
        "evaluation_kinematic_distance": float(np.mean(evaluation_pairs.distance)),
    }


def evaluate_outer_fold(
    r1: dict,
    r2: dict,
    calibration_r1: np.ndarray,
    calibration_r2: np.ndarray,
    evaluation_r1: np.ndarray,
    evaluation_r2: np.ndarray,
    *,
    fit_seed: int,
    random_reps: int,
    retention_fractions: tuple[float, ...],
    selectors: tuple[str, ...],
) -> list[dict]:
    """Fit once on calibration and evaluate all locked dose conditions."""
    if np.any(calibration_r1 & evaluation_r1) or np.any(calibration_r2 & evaluation_r2):
        raise ValueError("Calibration and evaluation masks must be disjoint")
    alignment = fit_calibration_alignment(
        r1, r2, calibration_r1, calibration_r2, fit_seed
    )
    alignment_id = _alignment_hash(alignment)
    activity_r1 = alignment.transform_train(r1["Y"])
    activity_r2 = alignment.transform_target(r2["Y"])
    signal_r1, residual_r1 = fit_kinematic_signal(
        activity_r1, r1["Kin"], calibration_r1
    )
    signal_r2, residual_r2 = fit_kinematic_signal(
        activity_r2, r2["Kin"], calibration_r2
    )

    scaler = fit_match_scaler(
        r1["Kin"], r1["meta"], calibration_r1,
        r2["Kin"], r2["meta"], calibration_r2,
    )
    calibration_pairs = match_trial_sets(
        r1["Kin"], r1["meta"], calibration_r1,
        r2["Kin"], r2["meta"], calibration_r2,
        scaler,
    )
    evaluation_pairs = match_trial_sets(
        r1["Kin"], r1["meta"], evaluation_r1,
        r2["Kin"], r2["meta"], evaluation_r2,
        scaler,
    )
    if len(calibration_pairs) < MIN_CALIBRATION_PAIRS:
        raise ValueError("Too few calibration trial pairs")
    if len(evaluation_pairs) < MIN_EVALUATION_PAIRS:
        raise ValueError("Too few evaluation trial pairs")

    residual_trajectories_r1 = trial_trajectory_map(
        residual_r1,
        r1["meta"],
        np.concatenate([calibration_pairs.r1_trials, evaluation_pairs.r1_trials]),
    )
    residual_trajectories_r2 = trial_trajectory_map(
        residual_r2,
        r2["meta"],
        np.concatenate([calibration_pairs.r2_trials, evaluation_pairs.r2_trials]),
    )
    calibration_v1, calibration_v2, calibration_ratio = paired_residual_variance(
        residual_trajectories_r1, residual_trajectories_r2, calibration_pairs
    )
    high_side = "r2" if calibration_v2 >= calibration_v1 else "r1"
    quiet_high, gate_diagnostics = fit_neural_only_gate(
        activity_r1,
        activity_r2,
        signal_r1,
        signal_r2,
        residual_r1,
        residual_r2,
        calibration_r1,
        calibration_r2,
        high_side,
    )

    if high_side == "r2":
        neural_scores = trial_change_scores(
            activity_r2,
            quiet_high,
            r2["meta"],
            np.concatenate([calibration_pairs.r2_trials, evaluation_pairs.r2_trials]),
        )
    else:
        neural_scores = trial_change_scores(
            activity_r1,
            quiet_high,
            r1["meta"],
            np.concatenate([calibration_pairs.r1_trials, evaluation_pairs.r1_trials]),
        )

    # Keep the primary path structurally neural-only: held-out residual scores
    # (and therefore held-out movement) are not even constructed unless the
    # explicitly target-conditioned oracle sensitivity is requested.
    score_lookup = {"neural_only": neural_scores}
    oracle_scores = None
    if "oracle_residual" in selectors:
        if high_side == "r2":
            oracle_scores = trial_oracle_residual_scores(
                residual_r2,
                r2["meta"],
                np.concatenate(
                    [calibration_pairs.r2_trials, evaluation_pairs.r2_trials]
                ),
            )
        else:
            oracle_scores = trial_oracle_residual_scores(
                residual_r1,
                r1["meta"],
                np.concatenate(
                    [calibration_pairs.r1_trials, evaluation_pairs.r1_trials]
                ),
            )
        score_lookup["oracle_residual"] = oracle_scores
    rows = []
    for selector_index, selector in enumerate(selectors):
        if selector not in VALID_SELECTORS:
            raise ValueError(f"Unknown selector: {selector}")
        calibration_scores = scores_for_pairs(
            calibration_pairs, score_lookup[selector], high_side
        )
        evaluation_scores = scores_for_pairs(
            evaluation_pairs, score_lookup[selector], high_side
        )
        score_correlation = (
            float(
                np.corrcoef(
                    scores_for_pairs(evaluation_pairs, neural_scores, high_side),
                    scores_for_pairs(evaluation_pairs, oracle_scores, high_side),
                )[0, 1]
            )
            if oracle_scores is not None and len(evaluation_pairs) > 2
            else np.nan
        )

        for dose_index, dose in enumerate(retention_fractions):
            cal_keep, eval_keep, cutoff = calibration_cutoff_selection(
                calibration_scores, evaluation_scores, dose
            )
            variance_calibration = calibration_pairs.subset(cal_keep)
            variance_evaluation = evaluation_pairs.subset(eval_keep)
            if (
                len(variance_calibration) < MIN_CALIBRATION_PAIRS
                or len(variance_evaluation) < MIN_EVALUATION_PAIRS
            ):
                rows.append({
                    "selector": selector,
                    "retention_fraction": dose,
                    "condition": "invalid_too_few_pairs",
                    "random_rep": -1,
                    "valid": False,
                    "high_variance_side": high_side,
                    "calibration_cutoff": cutoff,
                    "n_calibration_pairs_eligible": len(calibration_pairs),
                    "n_evaluation_pairs_eligible": len(evaluation_pairs),
                    "n_calibration_pairs_retained": len(variance_calibration),
                    "n_evaluation_pairs_retained": len(variance_evaluation),
                    "calibration_fraction_actual": len(variance_calibration) / len(calibration_pairs),
                    "evaluation_fraction_actual": len(variance_evaluation) / len(evaluation_pairs),
                    "calibration_residual_variance_ratio_all": calibration_ratio,
                    "cca_fit_id": alignment_id,
                    "gate_fit_id": gate_diagnostics["gate_fit_id"],
                    "gate_uses_evaluation_kinematics": selector == "oracle_residual",
                    "matching_uses_evaluation_kinematics": True,
                })
                continue

            common = {
                "selector": selector,
                "retention_fraction": dose,
                "valid": True,
                "high_variance_side": high_side,
                "calibration_cutoff": cutoff,
                "n_calibration_pairs_eligible": len(calibration_pairs),
                "n_evaluation_pairs_eligible": len(evaluation_pairs),
                "calibration_fraction_actual": len(variance_calibration) / len(calibration_pairs),
                "evaluation_fraction_actual": len(variance_evaluation) / len(evaluation_pairs),
                "calibration_residual_variance_ratio_all": calibration_ratio,
                "cca_fit_id": alignment_id,
                "gate_fit_id": gate_diagnostics["gate_fit_id"],
                "gate_uses_evaluation_kinematics": selector == "oracle_residual",
                "matching_uses_evaluation_kinematics": True,
                "neural_oracle_score_corr_evaluation": score_correlation,
                **gate_diagnostics,
            }
            if dose == 1.0:
                decoded = decode_pair_sets(
                    r1, r2, activity_r1, activity_r2,
                    calibration_pairs, evaluation_pairs,
                )
                rows.append({
                    **common,
                    "condition": "all",
                    "random_rep": -1,
                    **decoded,
                    **_row_diagnostics(
                        calibration_pairs,
                        evaluation_pairs,
                        residual_trajectories_r1,
                        residual_trajectories_r2,
                    ),
                })
                continue

            decoded_variance = decode_pair_sets(
                r1, r2, activity_r1, activity_r2,
                variance_calibration, variance_evaluation,
            )
            rows.append({
                **common,
                "condition": "variance_both",
                "random_rep": -1,
                **decoded_variance,
                **_row_diagnostics(
                    variance_calibration,
                    variance_evaluation,
                    residual_trajectories_r1,
                    residual_trajectories_r2,
                ),
            })

            for random_rep in range(random_reps):
                random_seed = (
                    fit_seed
                    + selector_index * 10_000_000
                    + dose_index * 100_000
                    + random_rep * 1000
                )
                rng = np.random.default_rng(random_seed)
                random_calibration = calibration_pairs.subset(
                    random_pair_indices(
                        len(calibration_pairs), len(variance_calibration), rng
                    )
                )
                random_evaluation = evaluation_pairs.subset(
                    random_pair_indices(
                        len(evaluation_pairs), len(variance_evaluation), rng
                    )
                )
                cells = factorial_pair_sets(
                    variance_calibration,
                    variance_evaluation,
                    random_calibration,
                    random_evaluation,
                )
                for condition in (
                    "random_both",
                    "variance_train_only",
                    "variance_eval_only",
                ):
                    cal_pairs, eval_pairs = cells[condition]
                    decoded = decode_pair_sets(
                        r1, r2, activity_r1, activity_r2,
                        cal_pairs, eval_pairs,
                    )
                    rows.append({
                        **common,
                        "condition": condition,
                        "random_rep": random_rep,
                        **decoded,
                        **_row_diagnostics(
                            cal_pairs,
                            eval_pairs,
                            residual_trajectories_r1,
                            residual_trajectories_r2,
                        ),
                    })
    return rows


def fold_effects(rows: pd.DataFrame) -> pd.DataFrame:
    """Contrast variance policies with same-fold equal-count random controls."""
    rows = rows.loc[rows["valid"] == True].copy()  # noqa: E712
    identifiers = [
        "r1",
        "r2",
        "r1_session",
        "r2_session",
        "selector",
        "retention_fraction",
        "repeat",
        "fold",
    ]
    metrics = [
        "forward_corr",
        "reverse_corr",
        "gap_corr",
        "gap_fisher_z",
        "own_r1_corr",
        "own_r2_corr",
        "calibration_residual_variance_ratio",
        "evaluation_residual_variance_ratio",
        "calibration_kinematic_distance",
        "evaluation_kinematic_distance",
        "calibration_fraction_actual",
        "evaluation_fraction_actual",
    ]
    factorial_metrics = [
        "forward_corr",
        "reverse_corr",
        "gap_corr",
        "gap_fisher_z",
        "own_r1_corr",
        "own_r2_corr",
    ]
    effects = []
    for keys, group in rows.groupby(identifiers, sort=True):
        record = dict(zip(identifiers, keys))
        dose = float(record["retention_fraction"])
        if dose == 1.0:
            selected = group[group["condition"] == "all"].iloc[0]
            for metric in metrics:
                record[f"selected_{metric}"] = float(selected[metric])
                record[f"random_{metric}"] = float(selected[metric])
                record[f"delta_{metric}"] = 0.0
            for metric in factorial_metrics:
                record[f"train_only_{metric}"] = float(selected[metric])
                record[f"eval_only_{metric}"] = float(selected[metric])
                record[f"delta_train_only_{metric}"] = 0.0
                record[f"delta_eval_only_{metric}"] = 0.0
                record[f"interaction_{metric}"] = 0.0
            record["n_random_reps"] = 0
            effects.append(record)
            continue
        selected_rows = group[group["condition"] == "variance_both"]
        random_rows = group[group["condition"] == "random_both"]
        train_rows = group[group["condition"] == "variance_train_only"]
        eval_rows = group[group["condition"] == "variance_eval_only"]
        if len(selected_rows) != 1 or min(len(random_rows), len(train_rows), len(eval_rows)) == 0:
            continue
        selected = selected_rows.iloc[0]
        for metric in metrics:
            selected_value = float(selected[metric])
            random_value = float(random_rows[metric].mean())
            record[f"selected_{metric}"] = selected_value
            record[f"random_{metric}"] = random_value
            record[f"delta_{metric}"] = selected_value - random_value
        for metric in factorial_metrics:
            selected_value = float(selected[metric])
            random_value = float(random_rows[metric].mean())
            train_value = float(train_rows[metric].mean())
            eval_value = float(eval_rows[metric].mean())
            record[f"train_only_{metric}"] = train_value
            record[f"eval_only_{metric}"] = eval_value
            record[f"delta_train_only_{metric}"] = train_value - random_value
            record[f"delta_eval_only_{metric}"] = eval_value - random_value
            record[f"interaction_{metric}"] = (
                selected_value - train_value - eval_value + random_value
            )
        record["n_random_reps"] = int(random_rows["random_rep"].nunique())
        effects.append(record)
    return pd.DataFrame(effects)


def summarize_effects(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identifiers = ["r1", "r2", "r1_session", "r2_session", "selector", "retention_fraction"]
    numeric = [
        column
        for column in effects.columns
        if column not in identifiers + ["repeat", "fold"]
        and pd.api.types.is_numeric_dtype(effects[column])
    ]
    pairs = effects.groupby(identifiers, as_index=False)[numeric].mean()
    days = pairs.groupby(
        ["r2", "r2_session", "selector", "retention_fraction"], as_index=False
    )[numeric].mean()

    endpoint_columns = [
        "delta_gap_fisher_z",
        "delta_forward_corr",
        "delta_reverse_corr",
        "delta_own_r1_corr",
        "delta_own_r2_corr",
        "delta_train_only_gap_fisher_z",
        "delta_eval_only_gap_fisher_z",
        "interaction_gap_fisher_z",
        "delta_train_only_forward_corr",
        "delta_eval_only_forward_corr",
        "interaction_forward_corr",
        "delta_train_only_reverse_corr",
        "delta_eval_only_reverse_corr",
        "interaction_reverse_corr",
    ]
    summary_rows = []
    for (selector, dose), group in days.groupby(
        ["selector", "retention_fraction"], sort=True
    ):
        row = {
            "selector": selector,
            "retention_fraction": dose,
            "n_r2_days": int(group["r2"].nunique()),
            "n_session_pairs": int(
                pairs.loc[
                    (pairs["selector"] == selector)
                    & (pairs["retention_fraction"] == dose)
                ].shape[0]
            ),
        }
        for column in [
            "selected_forward_corr",
            "random_forward_corr",
            "selected_reverse_corr",
            "random_reverse_corr",
            "selected_gap_corr",
            "random_gap_corr",
            "selected_gap_fisher_z",
            "random_gap_fisher_z",
            "selected_evaluation_residual_variance_ratio",
            "random_evaluation_residual_variance_ratio",
            "selected_evaluation_fraction_actual",
        ] + endpoint_columns:
            values = group[column].to_numpy(dtype=float)
            row[column] = float(np.nanmean(values))
            row[f"{column}_sd_r2_days"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
            )
            if column in endpoint_columns:
                row[f"{column}_r2_day_signflip_p"] = exact_sign_flip_p(values)
        summary_rows.append(row)
    return pairs, days, pd.DataFrame(summary_rows)


def validity_coverage(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit transferred-cutoff validity without counting condition duplicates.

    Every valid fold/dose has many rows because of the factorial cells and random
    repeats, whereas an invalid fold/dose has one marker row.  Collapse those
    rows first so coverage reflects independent outer-fold instances rather than
    the number of decoder fits.
    """
    fold_keys = [
        "r1",
        "r2",
        "r1_session",
        "r2_session",
        "selector",
        "retention_fraction",
        "repeat",
        "fold",
    ]
    folds = rows.groupby(fold_keys, as_index=False).agg(
        valid=("valid", "max"),
        high_variance_side=("high_variance_side", "first"),
    )
    folds["valid"] = folds["valid"].astype(bool)
    folds["high_side_is_r2"] = folds["high_variance_side"].eq("r2")

    def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        result = frame.groupby(keys, as_index=False).agg(
            n_fold_instances_total=("valid", "size"),
            n_fold_instances_valid=("valid", "sum"),
            fraction_folds_high_side_r2=("high_side_is_r2", "mean"),
        )
        result["valid_fold_fraction"] = (
            result["n_fold_instances_valid"] / result["n_fold_instances_total"]
        )
        return result

    pair_keys = [
        "r1",
        "r2",
        "r1_session",
        "r2_session",
        "selector",
        "retention_fraction",
    ]
    pair_coverage = aggregate(folds, pair_keys)
    day_coverage = aggregate(
        folds,
        ["r2", "r2_session", "selector", "retention_fraction"],
    )
    overall = aggregate(folds, ["selector", "retention_fraction"])
    pair_any = pair_coverage.assign(
        has_valid=pair_coverage["n_fold_instances_valid"].gt(0)
    ).groupby(["selector", "retention_fraction"], as_index=False).agg(
        n_session_pairs_total=("has_valid", "size"),
        n_session_pairs_with_valid=("has_valid", "sum"),
    )
    day_any = day_coverage.assign(
        has_valid=day_coverage["n_fold_instances_valid"].gt(0)
    ).groupby(["selector", "retention_fraction"], as_index=False).agg(
        n_r2_days_total=("has_valid", "size"),
        n_r2_days_with_valid=("has_valid", "sum"),
    )
    overall = overall.merge(
        pair_any, on=["selector", "retention_fraction"], how="left"
    ).merge(day_any, on=["selector", "retention_fraction"], how="left")
    return pair_coverage, day_coverage, overall


def make_figure(days: pd.DataFrame, output: Path, selector: str = PRIMARY_SELECTOR) -> None:
    frame = days[days["selector"] == selector].copy()
    if frame.empty:
        return
    doses = np.asarray(sorted(frame["retention_fraction"].unique(), reverse=True))
    colors = {r2: color for r2, color in zip(
        sorted(frame["r2"].unique()), ["#e74c3c", "#3498db", "#2ca02c"]
    )}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    def plot_day_lines(ax, column, *, zero=False, one=False):
        for r2, group in frame.groupby("r2"):
            values = group.set_index("retention_fraction")[column].reindex(doses)
            ax.plot(doses, values, marker="o", color=colors[r2], alpha=0.72, label=r2)
        mean = frame.groupby("retention_fraction")[column].mean().reindex(doses)
        ax.plot(doses, mean, marker="o", color="black", lw=2.5, label="R2-day mean")
        if zero:
            ax.axhline(0, color="0.45", ls="--", lw=1)
        if one:
            ax.axhline(1, color="0.45", ls="--", lw=1)
        ax.invert_xaxis()
        ax.set_xlabel("calibration retention fraction")
        ax.grid(axis="y", alpha=0.2)

    plot_day_lines(
        axes[0], "selected_evaluation_residual_variance_ratio", one=True
    )
    axes[0].set_ylabel("held-out residual variance R2 / R1")
    axes[0].set_title("A  Does the gate transfer?")

    for column, marker, label in [
        ("selected_forward_corr", "o", "R1→R2"),
        ("selected_reverse_corr", "s", "R2→R1"),
    ]:
        mean = frame.groupby("retention_fraction")[column].mean().reindex(doses)
        axes[1].plot(doses, mean, marker=marker, lw=2, label=label)
    axes[1].invert_xaxis()
    axes[1].set_xlabel("calibration retention fraction")
    axes[1].set_ylabel("held-out decode correlation")
    axes[1].set_title("B  Selected paired trials")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    for column, marker, label in [
        ("selected_gap_corr", "o", "variance selected"),
        ("random_gap_corr", "s", "random equal N"),
    ]:
        mean = frame.groupby("retention_fraction")[column].mean().reindex(doses)
        axes[2].plot(doses, mean, marker=marker, lw=2, label=label)
    axes[2].axhline(0, color="0.45", ls="--", lw=1)
    axes[2].invert_xaxis()
    axes[2].set_xlabel("calibration retention fraction")
    axes[2].set_ylabel("R2→R1 − R1→R2")
    axes[2].set_title("C  Directional gap")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)

    plot_day_lines(axes[3], "delta_gap_fisher_z", zero=True)
    axes[3].set_ylabel("selected − random gap (Fisher z)")
    axes[3].set_title("D  Primary equal-N contrast")
    handles, labels = axes[3].get_legend_handles_labels()
    axes[3].legend(handles, labels, frameon=False, fontsize=8)

    fig.suptitle(
        "Cross-fitted neural-only variance-removal dose response",
        fontsize=14,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_fraction_list(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(not 0 < value <= 1 for value in values):
        raise argparse.ArgumentTypeError("fractions must be comma-separated values in (0, 1]")
    if 1.0 not in values:
        raise argparse.ArgumentTypeError("fractions must include 1.0 as the all-trial baseline")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-index", type=int, choices=range(len(SESSIONS_R2)))
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--random-reps", type=int, default=N_RANDOM_REPS)
    parser.add_argument(
        "--retention-fractions",
        type=parse_fraction_list,
        default=RETAIN_FRACTIONS,
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=VALID_SELECTORS,
        default=[PRIMARY_SELECTOR],
    )
    parser.add_argument("--max-r1", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def run_shard(args: argparse.Namespace) -> Path:
    if args.r2_index is None:
        raise ValueError("--r2-index is required unless --summarize-only is used")
    repeats = 1 if args.smoke_test else args.repeats
    folds = 2 if args.smoke_test else args.folds
    random_reps = 1 if args.smoke_test else args.random_reps
    fractions = (1.0, 0.5) if args.smoke_test else tuple(args.retention_fractions)
    r1_sessions = list(SESSIONS_R1)
    if args.smoke_test:
        r1_sessions = r1_sessions[:1]
    elif args.max_r1 is not None:
        r1_sessions = r1_sessions[: args.max_r1]
    r2_session = SESSIONS_R2[args.r2_index]
    sessions = r1_sessions + [r2_session]
    print(
        f"Loading {len(sessions)} sessions for R2[{args.r2_index}] "
        f"{short_session(r2_session)} ...",
        flush=True,
    )
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    for data in cache.values():
        data["Kin"] = kinematic_design(data["X"], data["meta"])

    session_index = {
        session: index for index, session in enumerate(list(SESSIONS_R1) + list(SESSIONS_R2))
    }
    rows = []
    for r1_index, r1_session in enumerate(r1_sessions):
        r1, r2 = cache[r1_session], cache[r2_session]
        for repeat in range(repeats):
            folds_r1 = trial_folds(
                r1["meta"],
                folds,
                SEED + session_index[r1_session] * 100_000 + repeat * 1000,
            )
            folds_r2 = trial_folds(
                r2["meta"],
                folds,
                SEED + session_index[r2_session] * 100_000 + repeat * 1000,
            )
            for fold in range(folds):
                evaluation_r1 = folds_r1[fold]
                evaluation_r2 = folds_r2[fold]
                calibration_r1 = ~evaluation_r1
                calibration_r2 = ~evaluation_r2
                fit_seed = (
                    SEED
                    + args.r2_index * 10_000_000
                    + r1_index * 100_000
                    + repeat * 1000
                    + fold
                )
                fold_rows = evaluate_outer_fold(
                    r1,
                    r2,
                    calibration_r1,
                    calibration_r2,
                    evaluation_r1,
                    evaluation_r2,
                    fit_seed=fit_seed,
                    random_reps=random_reps,
                    retention_fractions=fractions,
                    selectors=tuple(args.selectors),
                )
                identifiers = {
                    "animal": "TS",
                    "target": TARGET,
                    "r1": short_session(r1_session),
                    "r2": short_session(r2_session),
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "repeat": repeat,
                    "fold": fold,
                }
                for row in fold_rows:
                    row.update(identifiers)
                    rows.append(row)
        print(
            f"[{r1_index + 1}/{len(r1_sessions)}] {short_session(r1_session)} "
            f"vs {short_session(r2_session)} complete",
            flush=True,
        )

    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"r2_{args.r2_index}" + ("_smoke" if args.smoke_test else "")
    output = args.output_dir / f"variance_dose_crossfit_{suffix}.csv"
    result.to_csv(output, index=False)
    config = {
        "analysis": "cross-fitted paired trial-removal dose response",
        "animal": "TS",
        "r2_index": args.r2_index,
        "r2_session": r2_session,
        "r1_sessions": r1_sessions,
        "repeats": repeats,
        "folds": folds,
        "random_reps": random_reps,
        "retention_fractions": list(fractions),
        "selectors": list(args.selectors),
        "primary_selector": PRIMARY_SELECTOR,
        "gate_uses_evaluation_kinematics": False,
        "matching_uses_evaluation_kinematics": True,
    }
    with (args.output_dir / f"variance_dose_config_{suffix}.json").open("w") as stream:
        json.dump(config, stream, indent=2)
    print(f"Saved {output} ({len(result)} rows)", flush=True)
    return output


def summarize_shards(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [output_dir / f"variance_dose_crossfit_r2_{index}.csv" for index in range(len(SESSIONS_R2))]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing full shard outputs: {missing}")
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    effects = fold_effects(rows)
    pairs, days, summary = summarize_effects(effects)
    coverage_pairs, coverage_days, coverage = validity_coverage(rows)
    summary = coverage.merge(
        summary,
        on=["selector", "retention_fraction"],
        how="left",
    )
    effects.to_csv(output_dir / "variance_dose_effects_folds.csv", index=False)
    pairs.to_csv(output_dir / "variance_dose_effects_pairs.csv", index=False)
    days.to_csv(output_dir / "variance_dose_effects_r2_days.csv", index=False)
    coverage_pairs.to_csv(
        output_dir / "variance_dose_validity_pairs.csv", index=False
    )
    coverage_days.to_csv(
        output_dir / "variance_dose_validity_r2_days.csv", index=False
    )
    summary.to_csv(output_dir / "variance_dose_summary.csv", index=False)
    make_figure(
        days,
        output_dir / "fig_variance_removal_dose_response_crossfit.png",
    )
    return effects, pairs, days, summary


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        _, _, _, summary = summarize_shards(args.output_dir)
        columns = [
            "selector",
            "retention_fraction",
            "valid_fold_fraction",
            "selected_evaluation_residual_variance_ratio",
            "random_evaluation_residual_variance_ratio",
            "selected_forward_corr",
            "selected_reverse_corr",
            "selected_gap_corr",
            "random_gap_corr",
            "delta_gap_fisher_z",
            "delta_train_only_gap_fisher_z",
            "delta_eval_only_gap_fisher_z",
            "interaction_gap_fisher_z",
            "delta_gap_fisher_z_r2_day_signflip_p",
        ]
        print(summary[columns].round(4).to_string(index=False))
        return
    run_shard(args)


if __name__ == "__main__":
    main()
