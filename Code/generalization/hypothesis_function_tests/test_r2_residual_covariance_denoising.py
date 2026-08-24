"""Tests for cross-fitted R2 residual-covariance denoising helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

WHY = Path(__file__).resolve().parents[1] / "analyses"
sys.path.insert(0, str(WHY))

from r2_residual_covariance_denoising import (  # noqa: E402
    covariance_transport,
    fit_neural_only_denoiser,
    isotropic_trace_transform,
    spd_power,
)


class ResidualCovarianceDenoisingTests(unittest.TestCase):
    def test_full_transport_matches_target_covariance(self):
        source = np.array([[4.0, 1.2], [1.2, 2.0]])
        target = np.array([[1.5, -0.3], [-0.3, 3.0]])
        transform, _ = covariance_transport(source, target, shrink_only=False)
        observed = transform.T @ source @ transform
        np.testing.assert_allclose(observed, target, atol=1e-9, rtol=1e-9)

    def test_directional_transform_never_amplifies_source_whitened_axes(self):
        source = np.array([[4.0, 1.2], [1.2, 2.0]])
        # One generalized target direction is quieter and the other is broader.
        target = np.array([[1.0, 0.0], [0.0, 5.0]])
        transform, eigenvalues = covariance_transport(source, target, shrink_only=True)
        observed = transform.T @ source @ transform
        inv = spd_power(source, -0.5)
        relative_observed = inv @ observed @ inv
        observed_eigenvalues = np.linalg.eigvalsh(relative_observed)

        self.assertLess(eigenvalues.min(), 1.0)
        self.assertGreater(eigenvalues.max(), 1.0)
        self.assertLessEqual(observed_eigenvalues.max(), 1.0 + 1e-9)
        self.assertGreaterEqual(observed_eigenvalues.min(), -1e-9)

    def test_isotropic_trace_transform_only_shrinks(self):
        source = np.eye(3) * 4.0
        target = np.eye(3)
        transform, scale = isotropic_trace_transform(source, target)
        self.assertAlmostEqual(scale, 0.5)
        np.testing.assert_allclose(transform.T @ source @ transform, target)

        transform, scale = isotropic_trace_transform(target, source)
        self.assertAlmostEqual(scale, 1.0)
        np.testing.assert_allclose(transform, np.eye(3))

    def test_neural_only_denoiser_generalizes_a_calibration_linear_map(self):
        rng = np.random.default_rng(8)
        activity = rng.normal(size=(500, 3))
        calibration = np.zeros(len(activity), dtype=bool)
        calibration[:300] = True
        weights = np.array(
            [[0.6, 0.1, 0.0], [-0.2, 0.8, 0.1], [0.0, 0.2, 0.5]]
        )
        target = activity[calibration] @ weights + np.array([1.0, -2.0, 0.5])
        transformed = fit_neural_only_denoiser(activity, target, calibration)
        expected = activity @ weights + np.array([1.0, -2.0, 0.5])
        np.testing.assert_allclose(transformed, expected, atol=3e-3, rtol=3e-3)


if __name__ == "__main__":
    unittest.main()
