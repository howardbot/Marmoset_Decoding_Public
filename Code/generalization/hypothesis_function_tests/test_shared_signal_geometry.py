"""Tests for phase-resolved shared-space signal/noise geometry."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Code.generalization.shared_signal_geometry import (
    covariance_spectrum_metrics,
    encoded_covariances,
    fit_generalized_axes,
    fit_movement_encoding,
    heldout_generalized_metrics,
    phase_geometry,
    phase_stack,
    pooled_spectrum_scaling,
    scaling_transform,
    transform_phase_components,
)


class TestSharedSignalGeometry(unittest.TestCase):
    def test_phase_decomposition_recovers_rank_one_signal(self):
        rng = np.random.default_rng(3)
        phase = np.linspace(0.0, 1.0, 30)
        signal = np.column_stack([np.sin(np.pi * phase), np.zeros_like(phase)])
        stack = signal[None, :, :] + 0.08 * rng.standard_normal((80, 30, 2))
        geometry = phase_geometry(stack)
        metrics = covariance_spectrum_metrics(geometry.signal_covariance)

        self.assertLess(metrics["effective_rank"], 1.05)
        self.assertGreater(metrics["top1_fraction"], 0.98)
        axes, _ = fit_generalized_axes(
            geometry.signal_covariance, geometry.noise_covariance
        )
        heldout = heldout_generalized_metrics(
            geometry.signal_covariance, geometry.noise_covariance, axes
        )
        self.assertGreater(heldout["snr_top1_fraction"], 0.98)

    def test_phase_stack_and_component_transform_preserve_trial_layout(self):
        rows = []
        values = []
        for trial, length in enumerate((7, 9, 11)):
            phase = np.linspace(0.0, 1.0, length)
            values.extend(np.column_stack([phase, 2.0 * phase]))
            rows.extend({"trial_number": trial} for _ in range(length))
        values = np.asarray(values)
        meta = pd.DataFrame(rows)
        mask = np.ones(len(meta), dtype=bool)
        stack = phase_stack(values, meta, mask, n_phase_bins=13)
        geometry = phase_geometry(stack)
        transformed = transform_phase_components(
            values,
            meta,
            mask,
            geometry.mean_curve,
            signal_transform=2.0 * np.eye(2),
        )

        self.assertEqual(stack.shape, (3, 13, 2))
        self.assertEqual(transformed.shape, values.shape)
        self.assertGreater(np.ptp(transformed[:, 0]), np.ptp(values[:, 0]))

    def test_spectrum_scaling_matches_diagonal_target(self):
        source = np.diag([4.0, 1.0])
        target = np.diag([1.0, 9.0])
        axes, scales = pooled_spectrum_scaling(source, target, ridge_fraction=0.0)
        transform = scaling_transform(axes, scales)
        observed = transform.T @ source @ transform

        np.testing.assert_allclose(observed, target, atol=1e-10)

    def test_movement_encoding_separates_predictable_signal(self):
        rng = np.random.default_rng(11)
        trials = np.repeat(np.arange(20), 15)
        meta = pd.DataFrame({"trial_number": trials})
        movement = rng.standard_normal((len(meta), 3))
        activity = np.column_stack([
            1.5 * movement[:, 0] - 0.4 * movement[:, 1],
            -0.7 * movement[:, 2],
        ])
        activity += 0.05 * rng.standard_normal(activity.shape)
        calibration = trials < 15
        evaluation = ~calibration
        model = fit_movement_encoding(
            activity, movement, meta, calibration
        )
        prediction = model.predict(movement, meta)
        signal, residual = encoded_covariances(
            activity, prediction, evaluation
        )

        self.assertGreater(np.trace(signal), np.trace(residual) * 50.0)


if __name__ == "__main__":
    unittest.main()
