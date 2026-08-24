"""Unit tests for the target-trial-nested CCA helpers."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Code.generalization.nested_cca import (
    fit_nested_alignment,
    fit_pca_projector,
    ridge_cca_rotations,
    resampled_trial_trajectories,
    three_way_trial_masks,
)


def trial_meta(n_trials: int, bins_per_trial: int) -> pd.DataFrame:
    return pd.DataFrame({
        "trial_number": np.repeat(np.arange(n_trials), bins_per_trial)
    })


class TestNestedCCA(unittest.TestCase):
    def test_pca_projector_uses_only_fit_rows(self):
        fit = np.arange(60.0).reshape(20, 3)
        projector = fit_pca_projector(fit, 2)
        evaluation = np.full((5, 3), 1_000_000.0)

        np.testing.assert_allclose(projector.mean, fit.mean(axis=0))
        self.assertEqual(projector.transform(evaluation).shape, (5, 2))

    def test_trial_resampling_keeps_trials_separate(self):
        meta = trial_meta(4, 5)
        values = np.column_stack([np.arange(20.0), np.arange(20.0) ** 2])
        trajectories = resampled_trial_trajectories(values, meta, 7)

        self.assertEqual(len(trajectories), 4)
        self.assertTrue(all(t.shape == (7, 2) for t in trajectories))

    def test_evaluation_values_cannot_change_fitted_alignment(self):
        rng = np.random.default_rng(4)
        train_meta = trial_meta(10, 6)
        calibration_meta = trial_meta(8, 6)
        train = rng.normal(size=(len(train_meta), 6))
        calibration = rng.normal(size=(len(calibration_meta), 5))

        model = fit_nested_alignment(
            train,
            train_meta,
            calibration,
            calibration_meta,
            n_components=3,
            n_phase_bins=8,
            rng=np.random.default_rng(10),
        )
        evaluation_a = rng.normal(size=(12, 5))
        evaluation_b = evaluation_a + 10_000.0

        transformed_a = model.transform_target(evaluation_a)
        transformed_b = model.transform_target(evaluation_b)
        self.assertFalse(np.allclose(transformed_a, transformed_b))
        np.testing.assert_allclose(model.target_pca.mean, calibration.mean(axis=0))

    def test_ridge_cca_rotations_are_finite_for_collinear_inputs(self):
        x = np.linspace(-1.0, 1.0, 100)
        train = np.column_stack([x, x, x ** 2])
        target = np.column_stack([2 * x, 2 * x, x ** 2 + 0.1])

        train_rotation, target_rotation = ridge_cca_rotations(train, target, ridge=0.1)

        self.assertEqual(train_rotation.shape, (3, 3))
        self.assertEqual(target_rotation.shape, (3, 3))
        self.assertTrue(np.isfinite(train_rotation).all())
        self.assertTrue(np.isfinite(target_rotation).all())

    def test_three_way_masks_are_trial_disjoint_and_exhaustive(self):
        meta = trial_meta(12, 4)
        masks = three_way_trial_masks(meta, seed=9)

        self.assertEqual(len(masks), 3)
        np.testing.assert_array_equal(np.sum(masks, axis=0), 1)
        trial_sets = [set(meta.loc[mask, "trial_number"]) for mask in masks]
        self.assertFalse(trial_sets[0] & trial_sets[1])
        self.assertFalse(trial_sets[0] & trial_sets[2])
        self.assertFalse(trial_sets[1] & trial_sets[2])

    def test_trial_average_mode_fits_and_transforms(self):
        rng = np.random.default_rng(12)
        train_meta = trial_meta(10, 6)
        target_meta = trial_meta(9, 6)
        train = rng.normal(size=(len(train_meta), 6))
        target = rng.normal(size=(len(target_meta), 5))
        model = fit_nested_alignment(
            train,
            train_meta,
            target,
            target_meta,
            n_components=3,
            n_phase_bins=8,
            rng=np.random.default_rng(13),
            alignment_mode="average",
        )

        self.assertEqual(model.transform_train(train).shape, (len(train), 3))
        self.assertEqual(model.transform_target(target).shape, (len(target), 3))


if __name__ == "__main__":
    unittest.main()
