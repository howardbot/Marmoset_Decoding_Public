"""Small, swappable Kalman implementation for decoder mechanism diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Mapping

import numpy as np
import pandas as pd

COMPONENT_NAMES = ("A", "W", "H", "Q")


@dataclass(frozen=True)
class KalmanComponents:
    A: np.ndarray
    W: np.ndarray
    H: np.ndarray
    Q: np.ndarray


def transition_indices(meta: pd.DataFrame | None, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return valid t,t+1 indices, optionally excluding trial boundaries."""
    if meta is None:
        return np.arange(n_samples - 1), np.arange(1, n_samples)
    if len(meta) != n_samples:
        raise ValueError("metadata and samples must have equal length")
    starts = []
    ends = []
    for indices in meta.groupby("trial_number", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        if len(indices) > 1:
            starts.append(indices[:-1])
            ends.append(indices[1:])
    if not starts:
        raise ValueError("no within-trial transitions available")
    return np.concatenate(starts), np.concatenate(ends)


def fit_kalman_components(
    activity: np.ndarray,
    state: np.ndarray,
    meta: pd.DataFrame | None = None,
) -> KalmanComponents:
    """Fit A/W/H/Q using the equations in Neural_Decoding's Kalman decoder."""
    activity = np.asarray(activity, dtype=float)
    state = np.asarray(state, dtype=float)
    if activity.ndim != 2 or state.ndim != 2 or len(activity) != len(state):
        raise ValueError("activity and state must be aligned two-dimensional arrays")
    A, W = fit_state_dynamics(state, meta)
    x = state.T
    z = activity.T
    H = z @ x.T @ np.linalg.pinv(x @ x.T)
    observation_residual = z - H @ x
    Q = observation_residual @ observation_residual.T / len(state)
    return KalmanComponents(A=A, W=W, H=H, Q=_symmetrize(Q))


def fit_state_dynamics(
    state: np.ndarray,
    meta: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the state transition and process covariance."""
    state = np.asarray(state, dtype=float)
    if state.ndim != 2:
        raise ValueError("state must be a two-dimensional array")
    source_indices, target_indices = transition_indices(meta, len(state))
    x1 = state[source_indices].T
    x2 = state[target_indices].T
    A = x2 @ x1.T @ np.linalg.pinv(x1 @ x1.T)
    state_residual = x2 - A @ x1
    W = state_residual @ state_residual.T / len(source_indices)
    return A, _symmetrize(W)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2


def hybrid_components(
    source: KalmanComponents,
    target: KalmanComponents,
    target_mask: int,
) -> KalmanComponents:
    """Replace components selected by a four-bit mask with target-day values."""
    if not 0 <= target_mask < 2 ** len(COMPONENT_NAMES):
        raise ValueError("target_mask must be a four-bit integer")
    values = {}
    for index, name in enumerate(COMPONENT_NAMES):
        owner = target if target_mask & (1 << index) else source
        values[name] = getattr(owner, name)
    return KalmanComponents(**values)


def mask_label(mask: int) -> str:
    names = [name for index, name in enumerate(COMPONENT_NAMES) if mask & (1 << index)]
    return "+".join(names) if names else "source"


def _gain_sequence(model: KalmanComponents, n_steps: int) -> list[np.ndarray]:
    """Precompute gains using the same innovation update as the source decoder."""
    n_state = model.A.shape[0]
    covariance = np.zeros((n_state, n_state))
    gains = []
    for _ in range(max(0, n_steps - 1)):
        prior = model.A @ covariance @ model.A.T + model.W
        innovation_covariance = model.H @ prior @ model.H.T + model.Q
        gain = np.linalg.solve(
            innovation_covariance.T, (prior @ model.H.T).T
        ).T
        covariance = (np.eye(n_state) - gain @ model.H) @ prior
        covariance = _symmetrize(covariance)
        gains.append(gain)
    return gains


def steady_state_gain(
    model: KalmanComponents,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, int, float]:
    """Iterate the Kalman covariance recursion to its steady-state gain.

    Returns the gain, number of covariance updates, and the final maximum
    elementwise gain change. The recursion matches ``_gain_sequence`` exactly.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 2:
        raise ValueError("max_iterations must be at least two")

    n_state = model.A.shape[0]
    covariance = np.zeros((n_state, n_state))
    previous_gain = None
    for iteration in range(1, max_iterations + 1):
        prior = model.A @ covariance @ model.A.T + model.W
        innovation_covariance = model.H @ prior @ model.H.T + model.Q
        gain = np.linalg.solve(
            innovation_covariance.T, (prior @ model.H.T).T
        ).T
        covariance = (np.eye(n_state) - gain @ model.H) @ prior
        covariance = _symmetrize(covariance)
        if previous_gain is not None:
            change = float(np.max(np.abs(gain - previous_gain)))
            scale = max(1.0, float(np.max(np.abs(gain))))
            if change <= tolerance * scale:
                return gain, iteration, change
        previous_gain = gain
    raise RuntimeError(
        f"steady-state Kalman gain did not converge in {max_iterations} iterations"
    )


def predict_kalman_sequence(
    model: KalmanComponents,
    activity: np.ndarray,
    initial_state: np.ndarray,
) -> np.ndarray:
    activity = np.asarray(activity, dtype=float)
    state = np.asarray(initial_state, dtype=float).copy()
    prediction = np.empty((len(activity), len(state)), dtype=float)
    prediction[0] = state
    for time, gain in enumerate(_gain_sequence(model, len(activity))):
        prior_state = model.A @ state
        state = prior_state + gain @ (activity[time + 1] - model.H @ prior_state)
        prediction[time + 1] = state
    return prediction


def predict_kalman_trials(
    model: KalmanComponents,
    activity: np.ndarray,
    state: np.ndarray,
    meta: pd.DataFrame,
) -> np.ndarray:
    """Predict each trial independently, initializing from its true first state."""
    if not (len(activity) == len(state) == len(meta)):
        raise ValueError("activity, state, and metadata must have equal length")
    prediction = np.full_like(np.asarray(state, dtype=float), np.nan)
    for indices in meta.groupby("trial_number", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        prediction[indices] = predict_kalman_sequence(
            model, np.asarray(activity)[indices], np.asarray(state)[indices[0]]
        )
    return prediction


def shapley_values(scores: Mapping[int, float]) -> dict[str, float]:
    """Allocate the source-to-target score change over A/W/H/Q exactly."""
    n_components = len(COMPONENT_NAMES)
    expected_masks = set(range(2 ** n_components))
    if set(scores) != expected_masks:
        raise ValueError("scores must contain all 16 component masks")
    result = {}
    denominator = factorial(n_components)
    for index, name in enumerate(COMPONENT_NAMES):
        bit = 1 << index
        contribution = 0.0
        for mask in expected_masks:
            if mask & bit:
                continue
            subset_size = int(mask.bit_count())
            weight = (
                factorial(subset_size)
                * factorial(n_components - subset_size - 1)
                / denominator
            )
            contribution += weight * (scores[mask | bit] - scores[mask])
        result[name] = float(contribution)
    return result
