"""Tests for scale-invariant task-predictive read-out helpers."""
from __future__ import annotations

import unittest

import numpy as np

from Code.generalization.readout_compactness import (
    cumulative_predictive_energy,
    fit_predictive_readout,
    fit_whitening,
    predictive_effective_rank,
)


class TestReadoutCompactness(unittest.TestCase):
    def test_whitening_round_trip_and_covariance(self):
        rng = np.random.default_rng(4)
        raw = rng.standard_normal((2000, 3)) @ np.array(
            [[1.0, 0.7, 0.2], [0.0, 1.3, 0.4], [0.0, 0.0, 0.6]]
        )
        transform = fit_whitening(raw, ridge=1e-10)
        white = transform.transform(raw)

        np.testing.assert_allclose(
            np.cov(white, rowvar=False), np.eye(3), atol=1e-7
        )
        np.testing.assert_allclose(
            transform.inverse_transform(white), raw, atol=1e-10
        )

    def test_rank_one_signal_has_concentrated_predictive_spectrum(self):
        rng = np.random.default_rng(9)
        neural = rng.standard_normal((3000, 6))
        latent = neural[:, 0] + 0.2 * neural[:, 1]
        movement = np.column_stack([latent, 2.0 * latent, -0.5 * latent])
        movement += 0.03 * rng.standard_normal(movement.shape)
        model = fit_predictive_readout(neural[:2400], movement[:2400])

        self.assertGreater(cumulative_predictive_energy(model.singular_values, 1), 0.95)
        self.assertLess(predictive_effective_rank(model.singular_values), 1.11)
        rank_one = model.predict(neural[2400:], rank=1)
        rank_three = model.predict(neural[2400:], rank=3)
        rank_one_error = np.mean(np.square(rank_one - movement[2400:]))
        rank_three_error = np.mean(np.square(rank_three - movement[2400:]))
        self.assertLess(rank_one_error, rank_three_error * 1.02)

    def test_invalid_prediction_rank_is_rejected(self):
        rng = np.random.default_rng(2)
        model = fit_predictive_readout(
            rng.standard_normal((50, 4)), rng.standard_normal((50, 3))
        )
        with self.assertRaises(ValueError):
            model.predict(rng.standard_normal((5, 4)), rank=4)


if __name__ == "__main__":
    unittest.main()
