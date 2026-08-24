"""Phase-resolved signal/noise geometry helpers for shared neural spaces."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseGeometry:
    mean_curve: np.ndarray
    signal_covariance: np.ndarray
    noise_covariance: np.ndarray


@dataclass(frozen=True)
class MovementEncoding:
    design_mean: np.ndarray
    design_scale: np.ndarray
    activity_mean: np.ndarray
    weights: np.ndarray

    def predict(self, movement: np.ndarray, meta) -> np.ndarray:
        design = kinematic_design(movement, meta)
        standardized = (design - self.design_mean) / self.design_scale
        return self.activity_mean + standardized @ self.weights


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return (matrix + matrix.T) / 2.0


def _covariance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("covariance input must have at least two rows")
    centered = values - values.mean(axis=0, keepdims=True)
    return _symmetric(centered.T @ centered / (len(values) - 1))


def phase_stack(
    values: np.ndarray,
    meta,
    mask: np.ndarray,
    n_phase_bins: int = 30,
) -> np.ndarray:
    """Interpolate every selected whole trial onto a common phase grid."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or len(values) != len(meta) or len(mask) != len(meta):
        raise ValueError("values, metadata, and mask must align")
    target_phase = np.linspace(0.0, 1.0, n_phase_bins)
    trajectories = []
    selected_meta = meta.loc[mask]
    for _, labels in selected_meta.groupby("trial_number", sort=True).groups.items():
        indices = meta.index.get_indexer(labels)
        if len(indices) < 3:
            continue
        source_phase = np.linspace(0.0, 1.0, len(indices))
        trajectories.append(np.column_stack([
            np.interp(target_phase, source_phase, values[indices, dim])
            for dim in range(values.shape[1])
        ]))
    if len(trajectories) < 2:
        raise ValueError("at least two usable trials are required")
    return np.stack(trajectories)


def phase_geometry(stack: np.ndarray) -> PhaseGeometry:
    """Separate a phase-stacked response into repeatable signal and residual noise."""
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3 or stack.shape[0] < 2 or stack.shape[1] < 2:
        raise ValueError("stack must be trials x phase x dimensions")
    mean_curve = stack.mean(axis=0)
    signal_covariance = _covariance(mean_curve)
    residuals = (stack - mean_curve[None, :, :]).reshape(-1, stack.shape[-1])
    noise_covariance = _covariance(residuals)
    return PhaseGeometry(mean_curve, signal_covariance, noise_covariance)


def kinematic_design(movement: np.ndarray, meta) -> np.ndarray:
    """Build position, within-trial velocity, and speed predictors."""
    movement = np.asarray(movement, dtype=float)
    if movement.ndim != 2 or len(movement) != len(meta):
        raise ValueError("movement and metadata must align")
    velocity = np.zeros_like(movement)
    for indices in meta.groupby("trial_number", sort=False).indices.values():
        indices = np.asarray(indices, dtype=int)
        if len(indices) >= 2:
            velocity[indices] = np.gradient(movement[indices], axis=0)
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    return np.column_stack([movement, velocity, speed])


def fit_movement_encoding(
    activity: np.ndarray,
    movement: np.ndarray,
    meta,
    calibration: np.ndarray,
    ridge: float = 1e-2,
) -> MovementEncoding:
    """Fit calibration-only movement-to-neural encoding in a shared space."""
    activity = np.asarray(activity, dtype=float)
    calibration = np.asarray(calibration, dtype=bool)
    design = kinematic_design(movement, meta)
    design_mean = design[calibration].mean(axis=0)
    design_scale = design[calibration].std(axis=0)
    design_scale = np.where(design_scale > 1e-8, design_scale, 1.0)
    standardized = (design - design_mean) / design_scale
    activity_mean = activity[calibration].mean(axis=0)
    predictors = standardized[calibration]
    response = activity[calibration] - activity_mean
    weights = np.linalg.solve(
        predictors.T @ predictors + ridge * np.eye(predictors.shape[1]),
        predictors.T @ response,
    )
    return MovementEncoding(
        design_mean=design_mean,
        design_scale=design_scale,
        activity_mean=activity_mean,
        weights=weights,
    )


def encoded_covariances(
    activity: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    activity = np.asarray(activity, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if activity.shape != prediction.shape or len(mask) != len(activity):
        raise ValueError("activity, prediction, and mask must align")
    return _covariance(prediction[mask]), _covariance(
        activity[mask] - prediction[mask]
    )


def effective_rank(values: np.ndarray) -> float:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    denominator = float(np.square(values).sum())
    return float(np.square(values.sum()) / denominator) if denominator > 0 else 0.0


def covariance_spectrum_metrics(covariance: np.ndarray) -> dict[str, float]:
    eigenvalues = np.linalg.eigvalsh(_symmetric(covariance))[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    return {
        "power": total / len(eigenvalues),
        "effective_rank": effective_rank(eigenvalues),
        "top1_fraction": float(eigenvalues[0] / total) if total > 0 else 0.0,
    }


def fit_generalized_axes(
    signal_covariance: np.ndarray,
    noise_covariance: np.ndarray,
    ridge_fraction: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit calibration axes ordered by repeatable signal relative to trial noise."""
    signal_covariance = _symmetric(signal_covariance)
    noise_covariance = _symmetric(noise_covariance)
    dimension = signal_covariance.shape[0]
    if signal_covariance.shape != (dimension, dimension) or noise_covariance.shape != (
        dimension, dimension
    ):
        raise ValueError("signal and noise covariance shapes must agree")
    scale = max(float(np.trace(noise_covariance) / dimension), 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(
        noise_covariance + ridge_fraction * scale * np.eye(dimension)
    )
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    inverse_sqrt = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))
    ) @ eigenvectors.T
    whitened_signal = _symmetric(
        inverse_sqrt @ signal_covariance @ inverse_sqrt
    )
    snr, directions = np.linalg.eigh(whitened_signal)
    order = np.argsort(snr)[::-1]
    return inverse_sqrt @ directions[:, order], np.clip(snr[order], 0.0, None)


def heldout_generalized_metrics(
    signal_covariance: np.ndarray,
    noise_covariance: np.ndarray,
    calibration_axes: np.ndarray,
) -> dict[str, float]:
    axes = np.asarray(calibration_axes, dtype=float)
    signal = np.diag(axes.T @ _symmetric(signal_covariance) @ axes)
    noise = np.diag(axes.T @ _symmetric(noise_covariance) @ axes)
    ratios = np.clip(signal, 0.0, None) / np.clip(noise, 1e-12, None)
    total = float(ratios.sum())
    return {
        "snr_sum": total,
        "snr_effective_rank": effective_rank(ratios),
        "snr_top1_fraction": float(ratios[0] / total) if total > 0 else 0.0,
        "snr_axis1": float(ratios[0]),
    }


def mean_curve_at_rows(mean_curve: np.ndarray, meta, mask: np.ndarray) -> np.ndarray:
    """Interpolate a phase mean back to each selected trial's original bins."""
    mean_curve = np.asarray(mean_curve, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    selected = np.flatnonzero(mask)
    result = np.empty((len(selected), mean_curve.shape[1]), dtype=float)
    position = {index: offset for offset, index in enumerate(selected)}
    source_phase = np.linspace(0.0, 1.0, len(mean_curve))
    selected_meta = meta.loc[mask]
    for _, labels in selected_meta.groupby("trial_number", sort=False).groups.items():
        indices = meta.index.get_indexer(labels)
        target_phase = np.linspace(0.0, 1.0, len(indices))
        interpolated = np.column_stack([
            np.interp(target_phase, source_phase, mean_curve[:, dim])
            for dim in range(mean_curve.shape[1])
        ])
        for index, row in zip(indices, interpolated):
            result[position[index]] = row
    return result


def pooled_spectrum_scaling(
    source_covariance: np.ndarray,
    target_covariance: np.ndarray,
    ridge_fraction: float = 1e-3,
    scale_bounds: tuple[float, float] = (0.25, 4.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled axes and source-to-target variance scale factors."""
    source_covariance = _symmetric(source_covariance)
    target_covariance = _symmetric(target_covariance)
    pooled = _symmetric((source_covariance + target_covariance) / 2.0)
    _, axes = np.linalg.eigh(pooled)
    axes = axes[:, ::-1]
    source_variance = np.diag(axes.T @ source_covariance @ axes)
    target_variance = np.diag(axes.T @ target_covariance @ axes)
    ridge = ridge_fraction * max(float(np.trace(pooled) / len(pooled)), 1e-12)
    scales = np.sqrt(
        (np.clip(target_variance, 0.0, None) + ridge)
        / (np.clip(source_variance, 0.0, None) + ridge)
    )
    return axes, np.clip(scales, *scale_bounds)


def scaling_transform(axes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if axes.shape != (len(scales), len(scales)):
        raise ValueError("axes and scales must describe a square space")
    return (axes * scales[None, :]) @ axes.T


def random_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    return basis


def transform_phase_components(
    activity: np.ndarray,
    meta,
    mask: np.ndarray,
    mean_curve: np.ndarray,
    signal_transform: np.ndarray | None = None,
    noise_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Transform phase signal and residual noise separately on selected trials."""
    activity = np.asarray(activity, dtype=float)
    dimension = activity.shape[1]
    signal_transform = (
        np.eye(dimension) if signal_transform is None else signal_transform
    )
    noise_transform = np.eye(dimension) if noise_transform is None else noise_transform
    phase_mean = mean_curve_at_rows(mean_curve, meta, mask)
    global_mean = np.asarray(mean_curve).mean(axis=0)
    selected = activity[mask]
    signal = phase_mean - global_mean
    residual = selected - phase_mean
    transformed = activity.copy()
    transformed[mask] = (
        global_mean + signal @ signal_transform + residual @ noise_transform
    )
    return transformed


def transform_encoded_components(
    activity: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    signal_transform: np.ndarray | None = None,
    residual_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Transform movement-predictable signal and encoding residual separately."""
    activity = np.asarray(activity, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    dimension = activity.shape[1]
    signal_transform = (
        np.eye(dimension) if signal_transform is None else signal_transform
    )
    residual_transform = (
        np.eye(dimension) if residual_transform is None else residual_transform
    )
    center = prediction[mask].mean(axis=0)
    transformed = activity.copy()
    transformed[mask] = (
        center
        + (prediction[mask] - center) @ signal_transform
        + (activity[mask] - prediction[mask]) @ residual_transform
    )
    return transformed
