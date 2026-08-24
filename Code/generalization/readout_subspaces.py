"""Pure linear-algebra helpers for cross-fitted read-out subspace tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReadoutSubspaces:
    shared: np.ndarray
    private_a: np.ndarray
    private_b: np.ndarray
    union: np.ndarray
    cosines: np.ndarray


def orthonormal_basis(values: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be two-dimensional")
    if values.shape[1] == 0:
        return np.empty((values.shape[0], 0))
    left, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] == 0:
        return np.empty((values.shape[0], 0))
    rank = int(np.sum(singular_values > tolerance * singular_values[0]))
    return left[:, :rank]


def readout_basis(weights: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """Orthonormal basis for the neural directions used by a linear read-out."""
    return orthonormal_basis(np.asarray(weights, dtype=float), tolerance)


def principal_readout_subspaces(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    cosine_threshold: float,
) -> ReadoutSubspaces:
    """Split two potent spaces into shared bisectors and weakly aligned directions."""
    basis_a = orthonormal_basis(basis_a)
    basis_b = orthonormal_basis(basis_b)
    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError("read-out bases must use the same ambient space")
    if not 0 <= cosine_threshold <= 1:
        raise ValueError("cosine_threshold must be in [0, 1]")

    cross = basis_a.T @ basis_b
    left, cosines, right_t = np.linalg.svd(cross, full_matrices=True)
    principal_a = basis_a @ left
    principal_b = basis_b @ right_t.T
    n_pairs = min(basis_a.shape[1], basis_b.shape[1])
    paired_cosines = cosines[:n_pairs]
    shared_mask = paired_cosines > cosine_threshold

    if np.any(shared_mask):
        bisectors = (
            principal_a[:, :n_pairs][:, shared_mask]
            + principal_b[:, :n_pairs][:, shared_mask]
        )
        shared = orthonormal_basis(bisectors)
    else:
        shared = np.empty((basis_a.shape[0], 0))

    cosine_a = np.zeros(basis_a.shape[1])
    cosine_b = np.zeros(basis_b.shape[1])
    cosine_a[:n_pairs] = paired_cosines
    cosine_b[:n_pairs] = paired_cosines
    private_a = principal_a[:, cosine_a <= cosine_threshold]
    private_b = principal_b[:, cosine_b <= cosine_threshold]
    union = orthonormal_basis(np.column_stack([basis_a, basis_b]))
    return ReadoutSubspaces(
        shared=shared,
        private_a=private_a,
        private_b=private_b,
        union=union,
        cosines=paired_cosines,
    )


def orthogonal_complement(basis: np.ndarray, n_features: int | None = None) -> np.ndarray:
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError("basis must be two-dimensional")
    ambient = basis.shape[0] if n_features is None else n_features
    if basis.shape[0] != ambient:
        raise ValueError("basis and ambient dimension disagree")
    basis = orthonormal_basis(basis)
    if basis.shape[1] == 0:
        return np.eye(ambient)
    _, _, right_t = np.linalg.svd(basis.T, full_matrices=True)
    return right_t[basis.shape[1]:].T


def random_subspace_within(
    container: np.ndarray,
    rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a Haar-like random subspace inside an orthonormal container."""
    container = orthonormal_basis(container)
    if not 0 <= rank <= container.shape[1]:
        raise ValueError("rank exceeds the container dimension")
    if rank == 0:
        return np.empty((container.shape[0], 0))
    rotation, _ = np.linalg.qr(rng.standard_normal((container.shape[1], rank)))
    return container @ rotation[:, :rank]
