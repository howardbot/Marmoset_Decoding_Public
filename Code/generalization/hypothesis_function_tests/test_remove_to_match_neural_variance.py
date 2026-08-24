"""Unit tests for paired trial-removal neural-variance matching."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

WHY = Path(__file__).resolve().parents[1] / "analyses"
sys.path.insert(0, str(WHY))

from remove_to_match_neural_variance import (  # noqa: E402
    exact_sign_flip_p,
    exact_two_group_permutation_p,
    select_pairs_to_match_variance,
    trial_variance,
)


class RemoveToMatchTests(unittest.TestCase):
    def test_trial_variance_is_across_trials_at_each_phase_and_dimension(self):
        stack = np.array([[[0.0]], [[2.0]]])
        self.assertTrue(np.isclose(trial_variance(stack), 1.0))

    def test_pair_removal_reduces_variance_mismatch_and_preserves_pair_indices(self):
        rng = np.random.default_rng(4)
        n, phase, dim = 20, 8, 3
        r1 = rng.normal(size=(n, phase, dim))
        r2 = r1.copy()
        r2[-3:] += rng.normal(scale=10.0, size=(3, phase, dim))
        before = trial_variance(r2) / trial_variance(r1)
        keep = select_pairs_to_match_variance(
            r1, r2, min_fraction=0.5, min_trials=5, tolerance=0.05
        )
        after = trial_variance(r2[keep]) / trial_variance(r1[keep])

        self.assertLessEqual(len(keep), n)
        self.assertGreaterEqual(len(keep), 10)
        self.assertEqual(len(np.unique(keep)), len(keep))
        self.assertLess(abs(np.log(after)), abs(np.log(before)))
        self.assertFalse(set(range(n - 3, n)).issubset(set(keep)))

    def test_exact_small_sample_p_values(self):
        # With three observations in each group, only one of the 20 allocations
        # is at least as extreme in the prespecified greater direction.
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([10.0, 11.0, 12.0])
        self.assertTrue(
            np.isclose(exact_two_group_permutation_p(a, b, "greater"), 1 / 20)
        )
        self.assertTrue(
            np.isclose(exact_two_group_permutation_p(a, b, "two-sided"), 2 / 20)
        )

        # For three same-sign paired effects, two of eight sign allocations have
        # an absolute mean as large as observed.
        self.assertTrue(np.isclose(exact_sign_flip_p([1.0, 1.0, 1.0]), 2 / 8))


if __name__ == "__main__":
    unittest.main()
