"""Tests for phase-resolved private-direction helpers."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Code.generalization.private_direction_localization import (
    directional_target_scale,
    kinematic_features,
    phase_correlations,
    phase_scaled_mse,
    phase_stack,
)


class TestPrivateDirectionLocalization(unittest.TestCase):
    def test_directional_target_scale_uses_scored_session(self):
        r1 = np.array([1.0, 2.0])
        r2 = np.array([3.0, 4.0])

        np.testing.assert_array_equal(
            directional_target_scale("forward", r1, r2), r2
        )
        np.testing.assert_array_equal(
            directional_target_scale("reverse", r1, r2), r1
        )

    def test_velocity_does_not_cross_trial_boundaries(self):
        position = np.array([
            [0, 0, 0], [1, 0, 0], [2, 0, 0],
            [100, 0, 0], [101, 0, 0], [102, 0, 0],
        ], dtype=float)
        meta = pd.DataFrame({"trial_number": [1, 1, 1, 2, 2, 2]})

        features = kinematic_features(position, meta, bin_seconds=1.0)

        np.testing.assert_allclose(features[:, 3], 1.0)
        np.testing.assert_allclose(features[:, 6], 1.0)

    def test_phase_stack_preserves_trial_endpoints(self):
        values = np.array([[0], [1], [3], [10], [13], [14], [18]], dtype=float)
        meta = pd.DataFrame({"trial_number": [1, 1, 1, 2, 2, 2, 2]})

        stacked, trials = phase_stack(values, meta, n_phase=5)

        np.testing.assert_array_equal(trials, [1, 2])
        np.testing.assert_allclose(stacked[:, 0, 0], [0, 10])
        np.testing.assert_allclose(stacked[:, -1, 0], [3, 18])

    def test_phase_correlations_recover_perfect_prediction(self):
        truth = np.arange(60.0).reshape(10, 3, 2)

        result = phase_correlations(truth, 2 * truth + 3)

        np.testing.assert_allclose(result, 1.0)

    def test_phase_scaled_mse_uses_feature_scale(self):
        truth = np.zeros((4, 3, 2))
        prediction = np.ones_like(truth) * np.array([2.0, 4.0])

        result = phase_scaled_mse(truth, prediction, np.array([2.0, 2.0]))

        np.testing.assert_allclose(result[:, 0], 1.0)
        np.testing.assert_allclose(result[:, 1], 4.0)


if __name__ == "__main__":
    unittest.main()
