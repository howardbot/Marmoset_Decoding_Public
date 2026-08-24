"""Scale-invariant task-predictive read-out spectrum helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WhiteningTransform:
    mean: np.ndarray
    whitener: np.ndarray
    colorer: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return (values - self.mean) @ self.whitener

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return values @ self.colorer + self.mean


@dataclass(frozen=True)
class PredictiveReadout:
    neural_whitening: WhiteningTransform
    movement_whitening: WhiteningTransform
    left: np.ndarray
    singular_values: np.ndarray
    right_t: np.ndarray

    def predict(self, neural: np.ndarray, rank: int) -> np.ndarray:
        if not 1 <= rank <= len(self.singular_values):
            raise ValueError("rank exceeds the task-predictive spectrum")
        neural_white = self.neural_whitening.transform(neural)
        movement_white = (
            neural_white @ self.left[:, :rank]
            * self.singular_values[:rank]
        ) @ self.right_t[:rank]
        return self.movement_whitening.inverse_transform(movement_white)


def fit_whitening(values: np.ndarray, ridge: float = 1e-6) -> WhiteningTransform:
    """Fit a symmetric, trace-regularized whitening transform."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("values must be a two-dimensional array with at least two rows")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / (len(values) - 1)
    scale = float(np.trace(covariance) / covariance.shape[0])
    regularized = covariance + ridge * max(scale, 1e-12) * np.eye(covariance.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh((regularized + regularized.T) / 2.0)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    whitener = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    colorer = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    return WhiteningTransform(mean=mean, whitener=whitener, colorer=colorer)


def fit_predictive_readout(
    neural: np.ndarray,
    movement: np.ndarray,
    ridge: float = 1e-6,
) -> PredictiveReadout:
    """Fit a whitened reduced-rank neural-to-movement read-out."""
    neural = np.asarray(neural, dtype=float)
    movement = np.asarray(movement, dtype=float)
    if neural.ndim != 2 or movement.ndim != 2 or len(neural) != len(movement):
        raise ValueError("neural and movement must be aligned two-dimensional arrays")
    neural_whitening = fit_whitening(neural, ridge)
    movement_whitening = fit_whitening(movement, ridge)
    neural_white = neural_whitening.transform(neural)
    movement_white = movement_whitening.transform(movement)
    cross_covariance = neural_white.T @ movement_white / max(len(neural) - 1, 1)
    left, singular_values, right_t = np.linalg.svd(
        cross_covariance, full_matrices=False
    )
    return PredictiveReadout(
        neural_whitening=neural_whitening,
        movement_whitening=movement_whitening,
        left=left,
        singular_values=singular_values,
        right_t=right_t,
    )


def predictive_effective_rank(singular_values: np.ndarray) -> float:
    """Participation ratio of squared task-predictive singular values."""
    energy = np.square(np.asarray(singular_values, dtype=float))
    denominator = float(np.square(energy).sum())
    if denominator <= 0:
        return 0.0
    return float(np.square(energy.sum()) / denominator)


def cumulative_predictive_energy(singular_values: np.ndarray, rank: int) -> float:
    energy = np.square(np.asarray(singular_values, dtype=float))
    total = float(energy.sum())
    if not 1 <= rank <= len(energy):
        raise ValueError("rank exceeds the task-predictive spectrum")
    return float(energy[:rank].sum() / total) if total > 0 else 0.0
