"""Pure helpers for calibration-only Kalman observation-map decomposition."""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Mapping, Sequence

import numpy as np

try:
    from .readout_subspaces import orthonormal_basis
except ImportError:  # Direct script execution adds Code/generalization to sys.path.
    from readout_subspaces import orthonormal_basis


@dataclass(frozen=True)
class ObservationModel:
    H: np.ndarray
    Q: np.ndarray
    b: np.ndarray


def fit_observation_model(
    activity: np.ndarray,
    state: np.ndarray,
    affine: bool,
) -> ObservationModel:
    """Fit neural activity = b + H state and its residual covariance."""
    activity = np.asarray(activity, dtype=float)
    state = np.asarray(state, dtype=float)
    if activity.ndim != 2 or state.ndim != 2 or len(activity) != len(state):
        raise ValueError("activity and state must be aligned two-dimensional arrays")
    if affine:
        design = np.column_stack([state, np.ones(len(state))])
        coefficients = np.linalg.pinv(design) @ activity
        H = coefficients[:-1].T
        b = coefficients[-1]
    else:
        x = state.T
        z = activity.T
        H = z @ x.T @ np.linalg.pinv(x @ x.T)
        b = np.zeros(activity.shape[1])
    residual = activity - (state @ H.T + b)
    Q = residual.T @ residual / len(state)
    Q = (Q + Q.T) / 2
    return ObservationModel(H=H, Q=Q, b=b)


def split_observation_delta(
    source_H: np.ndarray,
    target_H: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split target-source H into a neural subspace and its orthogonal rest."""
    source_H = np.asarray(source_H, dtype=float)
    target_H = np.asarray(target_H, dtype=float)
    if source_H.shape != target_H.shape or source_H.ndim != 2:
        raise ValueError("source_H and target_H must share a two-dimensional shape")
    basis = orthonormal_basis(np.asarray(basis, dtype=float))
    if basis.shape[0] != source_H.shape[0]:
        raise ValueError("basis and observation maps use different neural dimensions")
    delta = target_H - source_H
    private_delta = basis @ (basis.T @ delta)
    rest_delta = delta - private_delta
    return private_delta, rest_delta


def centering_offset(H: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Offset making zero-mean activity correspond to the state's calibration mean."""
    H = np.asarray(H, dtype=float)
    state = np.asarray(state, dtype=float)
    if H.ndim != 2 or state.ndim != 2 or H.shape[1] != state.shape[1]:
        raise ValueError("H and state dimensions are incompatible")
    return -(H @ state.mean(axis=0))


def apply_column_mask(
    source_H: np.ndarray,
    delta_H: np.ndarray,
    mask: int,
) -> np.ndarray:
    """Add selected state-variable columns of delta_H to source_H."""
    source_H = np.asarray(source_H, dtype=float)
    delta_H = np.asarray(delta_H, dtype=float)
    if source_H.shape != delta_H.shape or source_H.ndim != 2:
        raise ValueError("source_H and delta_H must share a two-dimensional shape")
    n_columns = source_H.shape[1]
    if not 0 <= mask < 2 ** n_columns:
        raise ValueError("column mask exceeds the observation-map width")
    selected = np.array([bool(mask & (1 << index)) for index in range(n_columns)])
    return source_H + delta_H * selected[None, :]


def shapley_values_named(
    scores: Mapping[int, float],
    names: Sequence[str],
) -> dict[str, float]:
    """Allocate a complete factorial score change over arbitrary named parts."""
    names = tuple(names)
    n_components = len(names)
    expected_masks = set(range(2 ** n_components))
    if not names or set(scores) != expected_masks:
        raise ValueError("scores must contain the complete factorial for names")
    denominator = factorial(n_components)
    result = {}
    for index, name in enumerate(names):
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
