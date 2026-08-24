"""Pure helpers for target-trial-nested cross-session CCA.

The target session is split before fitting. Its PCA basis and CCA rotation are
estimated from calibration trials only; evaluation trials are accepted only by
``transform_target`` after the model has been fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import CCA


@dataclass(frozen=True)
class PCAProjector:
    mean: np.ndarray
    components: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return (values - self.mean) @ self.components


@dataclass(frozen=True)
class NestedCCAAlignment:
    train_pca: PCAProjector
    target_pca: PCAProjector
    train_rotation: np.ndarray
    target_rotation: np.ndarray
    train_cca_mean: np.ndarray
    target_cca_mean: np.ndarray

    def transform_train(self, values: np.ndarray) -> np.ndarray:
        projected = self.train_pca.transform(values)
        return (projected - self.train_cca_mean) @ self.train_rotation

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        projected = self.target_pca.transform(values)
        return (projected - self.target_cca_mean) @ self.target_rotation


def fit_pca_projector(values: np.ndarray, n_components: int) -> PCAProjector:
    """Fit an SVD PCA projector using only ``values``."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("PCA input must be a two-dimensional array with at least two rows")
    max_components = min(values.shape)
    if not 1 <= n_components <= max_components:
        raise ValueError(f"n_components={n_components} exceeds PCA rank bound {max_components}")
    mean = values.mean(axis=0)
    _, _, vt = np.linalg.svd(values - mean, full_matrices=False)
    return PCAProjector(mean=mean, components=vt[:n_components].T)


def three_way_trial_masks(meta, seed: int) -> list[np.ndarray]:
    """Return three disjoint, trial-grouped boolean masks."""
    trials = np.asarray(sorted(meta["trial_number"].unique()))
    if len(trials) < 9:
        raise ValueError("At least nine trials are required for a three-way split")
    shuffled = np.random.default_rng(seed).permutation(trials)
    groups = np.array_split(shuffled, 3)
    return [meta["trial_number"].isin(group).to_numpy() for group in groups]


def resampled_trial_trajectories(values: np.ndarray, meta, n_phase_bins: int) -> list[np.ndarray]:
    """Resample each trial to a common phase grid without averaging trials."""
    values = np.asarray(values, dtype=float)
    if len(values) != len(meta):
        raise ValueError("values and meta must have the same number of rows")
    target_phase = np.linspace(0.0, 1.0, n_phase_bins)
    trajectories = []
    for _, indices in meta.groupby("trial_number", sort=True).indices.items():
        indices = np.asarray(indices)
        if len(indices) < 3:
            continue
        source_phase = np.linspace(0.0, 1.0, len(indices))
        trajectories.append(np.column_stack([
            np.interp(target_phase, source_phase, values[indices, dim])
            for dim in range(values.shape[1])
        ]))
    return trajectories


def _inverse_sqrt(covariance: np.ndarray, ridge: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    scale = float(np.trace(covariance) / covariance.shape[0])
    regularized = covariance + ridge * max(scale, 1e-12) * np.eye(covariance.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh((regularized + regularized.T) / 2.0)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def ridge_cca_rotations(
    train_samples: np.ndarray,
    target_samples: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear ridge-CCA rotations with trace-scaled covariance penalties."""
    if ridge <= 0:
        raise ValueError("ridge must be positive for ridge_cca_rotations")
    train_centered = train_samples - train_samples.mean(axis=0)
    target_centered = target_samples - target_samples.mean(axis=0)
    denominator = max(len(train_samples) - 1, 1)
    train_cov = train_centered.T @ train_centered / denominator
    target_cov = target_centered.T @ target_centered / denominator
    cross_cov = train_centered.T @ target_centered / denominator
    train_whitener = _inverse_sqrt(train_cov, ridge)
    target_whitener = _inverse_sqrt(target_cov, ridge)
    left, _, right_t = np.linalg.svd(
        train_whitener @ cross_cov @ target_whitener,
        full_matrices=False,
    )
    return train_whitener @ left, target_whitener @ right_t.T


def fit_phase_matched_cca(
    train_pc: np.ndarray,
    train_meta,
    target_pc: np.ndarray,
    target_meta,
    *,
    n_components: int,
    n_phase_bins: int,
    rng: np.random.Generator,
    ridge: float = 0.0,
    max_iter: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit phase-matched CCA rotations to already projected session activity."""
    train_trials = resampled_trial_trajectories(train_pc, train_meta, n_phase_bins)
    target_trials = resampled_trial_trajectories(target_pc, target_meta, n_phase_bins)
    n_trials = min(len(train_trials), len(target_trials))
    if n_trials < 3:
        raise ValueError("At least three usable trials per session are required for CCA")

    train_order = rng.permutation(len(train_trials))[:n_trials]
    target_order = rng.permutation(len(target_trials))[:n_trials]
    fit_train = np.vstack([train_trials[i] for i in train_order])
    fit_target = np.vstack([target_trials[i] for i in target_order])
    train_mean = fit_train.mean(axis=0)
    target_mean = fit_target.mean(axis=0)

    if ridge > 0:
        train_rotation, target_rotation = ridge_cca_rotations(
            fit_train, fit_target, ridge
        )
    else:
        cca = CCA(n_components=n_components, scale=False, max_iter=max_iter)
        cca.fit(fit_train, fit_target)
        train_rotation = np.asarray(cca.x_rotations_, dtype=float)
        target_rotation = np.asarray(cca.y_rotations_, dtype=float)
    return train_rotation, target_rotation, train_mean, target_mean


def fit_trial_average_cca(
    train_pc: np.ndarray,
    train_meta,
    target_pc: np.ndarray,
    target_meta,
    *,
    n_components: int,
    n_phase_bins: int,
    ridge: float = 0.0,
    max_iter: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit CCA to one trial-averaged trajectory per session."""
    train_trials = resampled_trial_trajectories(train_pc, train_meta, n_phase_bins)
    target_trials = resampled_trial_trajectories(target_pc, target_meta, n_phase_bins)
    if not train_trials or not target_trials:
        raise ValueError("No usable trials for trial-averaged CCA")
    fit_train = np.mean(np.stack(train_trials), axis=0)
    fit_target = np.mean(np.stack(target_trials), axis=0)
    train_mean = fit_train.mean(axis=0)
    target_mean = fit_target.mean(axis=0)
    if ridge > 0:
        train_rotation, target_rotation = ridge_cca_rotations(
            fit_train, fit_target, ridge
        )
    else:
        cca = CCA(n_components=n_components, scale=False, max_iter=max_iter)
        cca.fit(fit_train, fit_target)
        train_rotation = np.asarray(cca.x_rotations_, dtype=float)
        target_rotation = np.asarray(cca.y_rotations_, dtype=float)
    return train_rotation, target_rotation, train_mean, target_mean


def fit_nested_alignment(
    train_neural: np.ndarray,
    train_meta,
    target_calibration_neural: np.ndarray,
    target_calibration_meta,
    *,
    n_components: int,
    n_phase_bins: int,
    rng: np.random.Generator,
    alignment_mode: str = "single_trial",
    ridge: float = 0.0,
    max_iter: int = 5000,
) -> NestedCCAAlignment:
    """Fit train/target PCA and phase-matched CCA without target evaluation data."""
    train_pca = fit_pca_projector(train_neural, n_components)
    target_pca = fit_pca_projector(target_calibration_neural, n_components)
    train_pc = train_pca.transform(train_neural)
    target_pc = target_pca.transform(target_calibration_neural)

    if alignment_mode == "single_trial":
        train_rotation, target_rotation, train_mean, target_mean = fit_phase_matched_cca(
            train_pc,
            train_meta,
            target_pc,
            target_calibration_meta,
            n_components=n_components,
            n_phase_bins=n_phase_bins,
            rng=rng,
            ridge=ridge,
            max_iter=max_iter,
        )
    elif alignment_mode == "average":
        train_rotation, target_rotation, train_mean, target_mean = fit_trial_average_cca(
            train_pc,
            train_meta,
            target_pc,
            target_calibration_meta,
            n_components=n_components,
            n_phase_bins=n_phase_bins,
            ridge=ridge,
            max_iter=max_iter,
        )
    else:
        raise ValueError(f"Unknown alignment_mode: {alignment_mode}")
    return NestedCCAAlignment(
        train_pca=train_pca,
        target_pca=target_pca,
        train_rotation=train_rotation,
        target_rotation=target_rotation,
        train_cca_mean=train_mean,
        target_cca_mean=target_mean,
    )
