"""Pure helpers for phase-resolved private-direction characterization."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "speed",
)


def directional_target_scale(
    direction: str,
    r1_scale: np.ndarray,
    r2_scale: np.ndarray,
) -> np.ndarray:
    """Return the scale of the session on which directional error is scored."""
    if direction == "forward":
        return np.asarray(r2_scale, dtype=float)
    if direction == "reverse":
        return np.asarray(r1_scale, dtype=float)
    raise ValueError("direction must be 'forward' or 'reverse'")


def kinematic_features(
    position: np.ndarray,
    meta: pd.DataFrame,
    bin_seconds: float = 0.03,
) -> np.ndarray:
    """Return position, within-trial finite-difference velocity, and speed."""
    position = np.asarray(position, dtype=float)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position must have shape (samples, 3)")
    if len(position) != len(meta) or bin_seconds <= 0:
        raise ValueError("metadata length and bin_seconds must be valid")
    velocity = np.zeros_like(position)
    for indices in meta.groupby("trial_number", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        if len(indices) > 1:
            velocity[indices] = np.gradient(
                position[indices], bin_seconds, axis=0
            )
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    return np.column_stack([position, velocity, speed])


def phase_stack(
    values: np.ndarray,
    meta: pd.DataFrame,
    n_phase: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each trial to a common normalized phase grid."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != len(meta) or n_phase < 2:
        raise ValueError("values, metadata, and n_phase are incompatible")
    target_phase = np.linspace(0.0, 1.0, n_phase)
    stacked = []
    trials = []
    for trial, indices in meta.groupby("trial_number", sort=False).indices.items():
        indices = np.asarray(indices, dtype=int)
        if len(indices) < 3:
            continue
        source_phase = np.linspace(0.0, 1.0, len(indices))
        stacked.append(np.column_stack([
            np.interp(target_phase, source_phase, values[indices, feature])
            for feature in range(values.shape[1])
        ]))
        trials.append(trial)
    if not stacked:
        raise ValueError("no trials have at least three samples")
    return np.stack(stacked), np.asarray(trials)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 4 or np.std(left[valid]) < 1e-12 or np.std(right[valid]) < 1e-12:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def phase_correlations(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Correlation across trials for every phase x feature cell."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError("truth and prediction must share (trials, phase, features)")
    result = np.full(truth.shape[1:], np.nan)
    for phase in range(truth.shape[1]):
        for feature in range(truth.shape[2]):
            result[phase, feature] = safe_correlation(
                truth[:, phase, feature], prediction[:, phase, feature]
            )
    return result


def phase_scaled_mse(
    truth: np.ndarray,
    prediction: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    """Mean squared error across trials, normalized by calibration scale."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    feature_scale = np.asarray(feature_scale, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError("truth and prediction must share (trials, phase, features)")
    if feature_scale.shape != (truth.shape[2],):
        raise ValueError("feature_scale must match the feature dimension")
    scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    return np.nanmean(((truth - prediction) / scale[None, None, :]) ** 2, axis=0)


def mean_trial_correlations(
    truth: np.ndarray,
    prediction: np.ndarray,
    meta: pd.DataFrame,
) -> np.ndarray:
    """Mean within-trial correlation for each feature."""
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 2 or len(truth) != len(meta):
        raise ValueError("truth, prediction, and metadata are incompatible")
    rows = []
    for indices in meta.groupby("trial_number", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        if len(indices) < 4:
            continue
        rows.append([
            safe_correlation(truth[indices, feature], prediction[indices, feature])
            for feature in range(truth.shape[1])
        ])
    return np.nanmean(rows, axis=0) if rows else np.full(truth.shape[1], np.nan)
