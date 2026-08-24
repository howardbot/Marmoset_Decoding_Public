"""Unit tests for Mahalanobis train-support filtering."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Code.generalization.analyses.coverage_controls import (
    MASKS,
    summarize_masks,
    support_to_train,
    target_residual,
)


class CoverageControlTests(unittest.TestCase):
    def test_support_mask_uses_training_distribution(self):
        rng = np.random.default_rng(4)
        train = rng.standard_normal((1000, 2))
        near = rng.standard_normal((300, 2))
        far = rng.standard_normal((100, 2)) + 8.0
        result = support_to_train(train, np.vstack([near, far]), ridge=1e-6)

        self.assertGreater(result["mask"][: len(near)].mean(), 0.90)
        self.assertEqual(int(result["mask"][len(near):].sum()), 0)
        self.assertAlmostEqual(
            result["threshold"],
            np.percentile(result["train_distance"], 95),
        )

    def test_target_residual_removes_linear_target_component(self):
        rng = np.random.default_rng(7)
        target = rng.standard_normal((500, 3))
        weights = rng.standard_normal((3, 5))
        activity = target @ weights + 0.05 * rng.standard_normal((500, 5))
        residual = target_residual(activity, target)

        design = np.column_stack([target, np.ones(len(target))])
        np.testing.assert_allclose(design.T @ residual, 0.0, atol=1e-10)

    def test_mask_summary_preserves_full_gap_denominator(self):
        rows = []
        for pair in range(3):
            row = {"target": "relative_position"}
            for name in MASKS:
                row[f"{name}_R1R2"] = 0.4 + 0.01 * pair
                row[f"{name}_R2R1"] = 0.5 + 0.01 * pair
                row[f"retained_R2_{name}"] = 1.0 if name == "full" else 0.9
                row[f"retained_R1_{name}"] = 1.0 if name == "full" else 0.8
            rows.append(row)

        summary = summarize_masks(pd.DataFrame(rows)).set_index("mask")
        self.assertAlmostEqual(summary.loc["full", "asymmetry"], 0.1)
        self.assertAlmostEqual(summary.loc["neural", "gap_retained"], 1.0)
        self.assertAlmostEqual(summary.loc["neural", "mean_retained_R2_test"], 0.9)


if __name__ == "__main__":
    unittest.main()
