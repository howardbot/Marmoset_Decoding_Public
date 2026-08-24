"""Unit tests for swappable Kalman component diagnostics."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Code.generalization.kalman_components import (
    KalmanComponents,
    fit_kalman_components,
    fit_state_dynamics,
    hybrid_components,
    predict_kalman_sequence,
    shapley_values,
    steady_state_gain,
    transition_indices,
)
from Code.generalization.h_observation_decomposition import (
    apply_column_mask,
    centering_offset,
    fit_observation_model,
    shapley_values_named,
    split_observation_delta,
)


class TestKalmanComponents(unittest.TestCase):
    def test_no_intercept_observation_fit_matches_legacy_components(self):
        rng = np.random.default_rng(19)
        state = rng.normal(size=(80, 3))
        activity = rng.normal(size=(80, 7))

        legacy = fit_kalman_components(activity, state)
        observation = fit_observation_model(activity, state, affine=False)

        np.testing.assert_allclose(observation.H, legacy.H, atol=1e-12)
        np.testing.assert_allclose(observation.Q, legacy.Q, atol=1e-12)
        np.testing.assert_array_equal(observation.b, 0.0)

    def test_affine_observation_fit_recovers_offset_and_slope(self):
        rng = np.random.default_rng(21)
        state = rng.normal(size=(100, 3))
        H = rng.normal(size=(6, 3))
        b = rng.normal(size=6)
        activity = state @ H.T + b

        observation = fit_observation_model(activity, state, affine=True)

        np.testing.assert_allclose(observation.H, H, atol=1e-12)
        np.testing.assert_allclose(observation.b, b, atol=1e-12)
        np.testing.assert_allclose(observation.Q, 0.0, atol=1e-24)

    def test_affine_offset_equals_centering_identity_for_zero_mean_activity(self):
        rng = np.random.default_rng(22)
        state = rng.normal(size=(120, 3)) + np.array([2.0, -1.0, 0.5])
        H = rng.normal(size=(5, 3))
        activity = (state - state.mean(axis=0)) @ H.T

        observation = fit_observation_model(activity, state, affine=True)

        np.testing.assert_allclose(
            observation.b, centering_offset(observation.H, state), atol=1e-12
        )

    def test_private_and_rest_deltas_reconstruct_target(self):
        source = np.zeros((5, 3))
        target = np.arange(15.0).reshape(5, 3)
        basis = np.eye(5)[:, [1, 3]]

        private, rest = split_observation_delta(source, target, basis)

        np.testing.assert_allclose(source + private + rest, target)
        np.testing.assert_allclose((np.eye(5) - basis @ basis.T) @ private, 0.0)
        np.testing.assert_allclose(basis.T @ rest, 0.0)

    def test_column_mask_full_endpoint_reconstructs_delta(self):
        source = np.arange(12.0).reshape(4, 3)
        delta = np.ones_like(source) * 2

        np.testing.assert_array_equal(apply_column_mask(source, delta, 0), source)
        np.testing.assert_array_equal(apply_column_mask(source, delta, 7), source + delta)
        expected = source.copy()
        expected[:, 1] += 2
        np.testing.assert_array_equal(apply_column_mask(source, delta, 2), expected)

    def test_named_shapley_recovers_three_additive_parts(self):
        effects = np.array([0.2, -0.1, 0.35])
        scores = {
            mask: float(sum(effects[index] for index in range(3) if mask & (1 << index)))
            for mask in range(8)
        }

        result = shapley_values_named(scores, ("x", "y", "z"))

        np.testing.assert_allclose(list(result.values()), effects)
        self.assertAlmostEqual(sum(result.values()), scores[7] - scores[0])

    def test_trial_aware_transitions_exclude_boundaries(self):
        meta = pd.DataFrame({"trial_number": [1, 1, 1, 2, 2]})
        source, target = transition_indices(meta, len(meta))

        np.testing.assert_array_equal(source, [0, 1, 3])
        np.testing.assert_array_equal(target, [1, 2, 4])

    def test_concatenated_transitions_include_every_adjacent_row(self):
        source, target = transition_indices(None, 5)

        np.testing.assert_array_equal(source, [0, 1, 2, 3])
        np.testing.assert_array_equal(target, [1, 2, 3, 4])

    def test_hybrid_replaces_only_selected_components(self):
        source = KalmanComponents(*[np.full((1, 1), value) for value in range(4)])
        target = KalmanComponents(*[np.full((1, 1), value + 10) for value in range(4)])

        hybrid = hybrid_components(source, target, (1 << 0) | (1 << 2))

        self.assertEqual(hybrid.A.item(), 10)
        self.assertEqual(hybrid.W.item(), 1)
        self.assertEqual(hybrid.H.item(), 12)
        self.assertEqual(hybrid.Q.item(), 3)

    def test_shapley_values_recover_additive_component_effects(self):
        effects = np.array([0.1, -0.2, 0.4, 0.05])
        scores = {
            mask: float(sum(effects[index] for index in range(4) if mask & (1 << index)))
            for mask in range(16)
        }

        result = shapley_values(scores)

        np.testing.assert_allclose(list(result.values()), effects)
        self.assertAlmostEqual(sum(result.values()), scores[15] - scores[0])

    def test_fit_and_predict_recover_noiseless_linear_system(self):
        rng = np.random.default_rng(4)
        A = np.array([[0.8, 0.1], [-0.05, 0.9]])
        H = np.array([[1.0, 0.2], [-0.3, 0.8], [0.4, -0.1]])
        state = np.empty((80, 2))
        state[0] = [0.4, -0.2]
        for index in range(1, len(state)):
            state[index] = A @ state[index - 1] + rng.normal(scale=0.01, size=2)
        activity = state @ H.T + rng.normal(scale=0.01, size=(len(state), 3))

        model = fit_kalman_components(activity, state)
        prediction = predict_kalman_sequence(model, activity, state[0])

        self.assertLess(np.mean((prediction - state) ** 2), 0.002)

    def test_state_dynamics_recovers_known_transition(self):
        A = np.array([[0.85, 0.05], [-0.1, 0.9]])
        state = np.empty((100, 2))
        state[0] = [0.6, -0.3]
        for index in range(1, len(state)):
            state[index] = A @ state[index - 1]

        fitted, covariance = fit_state_dynamics(state)

        np.testing.assert_allclose(fitted, A, atol=1e-10)
        np.testing.assert_allclose(covariance, 0.0, atol=1e-20)

    def test_steady_state_gain_matches_long_gain_sequence(self):
        model = KalmanComponents(
            A=np.array([[0.85, 0.05], [-0.03, 0.90]]),
            W=np.diag([0.04, 0.03]),
            H=np.array([[1.0, 0.2], [-0.3, 0.8], [0.2, -0.1]]),
            Q=np.diag([0.20, 0.15, 0.25]),
        )

        gain, iterations, change = steady_state_gain(model)
        long_sequence = __import__(
            "Code.generalization.kalman_components", fromlist=["_gain_sequence"]
        )._gain_sequence(model, iterations + 50)

        np.testing.assert_allclose(gain, long_sequence[-1], atol=1e-9)
        self.assertGreater(iterations, 1)
        self.assertLessEqual(change, 1e-10)

    def test_steady_state_gain_rejects_invalid_controls(self):
        model = KalmanComponents(*[np.eye(1) for _ in range(4)])

        with self.assertRaises(ValueError):
            steady_state_gain(model, tolerance=0)
        with self.assertRaises(ValueError):
            steady_state_gain(model, max_iterations=1)

    def test_steady_state_gain_and_ols_use_the_same_neural_subspace(self):
        rng = np.random.default_rng(23)
        state = rng.normal(size=(400, 3))
        observation = rng.normal(size=(12, 3))
        activity = state @ observation.T + rng.normal(
            scale=0.7, size=(len(state), observation.shape[0])
        )

        model = fit_kalman_components(activity, state)
        gain, _, _ = steady_state_gain(model)
        ols = np.linalg.lstsq(activity, state, rcond=None)[0]
        kalman_basis, _ = np.linalg.qr(gain.T)
        ols_basis, _ = np.linalg.qr(ols)
        cosines = np.linalg.svd(
            kalman_basis[:, :3].T @ ols_basis[:, :3], compute_uv=False
        )

        np.testing.assert_allclose(cosines, 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
