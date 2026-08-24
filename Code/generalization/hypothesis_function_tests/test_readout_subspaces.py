"""Unit tests for read-out subspace decomposition helpers."""
from __future__ import annotations

import unittest

import numpy as np

from Code.generalization.readout_subspaces import (
    orthogonal_complement,
    principal_readout_subspaces,
    random_subspace_within,
    readout_basis,
)
from Code.generalization.analyses.private_readout_crossfit_summary import (
    add_map_decomposition,
    add_subspace_contrasts,
    hierarchical_interval,
)


class TestReadoutSubspaces(unittest.TestCase):
    def test_principal_decomposition_recovers_known_shared_and_private_axes(self):
        eye = np.eye(4)
        basis_a = eye[:, [0, 1]]
        basis_b = eye[:, [0, 2]]

        result = principal_readout_subspaces(basis_a, basis_b, cosine_threshold=0.5)

        self.assertEqual(result.shared.shape, (4, 1))
        self.assertEqual(result.private_a.shape, (4, 1))
        self.assertEqual(result.private_b.shape, (4, 1))
        self.assertAlmostEqual(abs(result.shared[:, 0] @ eye[:, 0]), 1.0)
        self.assertAlmostEqual(abs(result.private_a[:, 0] @ eye[:, 1]), 1.0)
        self.assertAlmostEqual(abs(result.private_b[:, 0] @ eye[:, 2]), 1.0)

    def test_orthogonal_complement_removes_requested_directions(self):
        basis = np.eye(5)[:, :2]
        complement = orthogonal_complement(basis)

        self.assertEqual(complement.shape, (5, 3))
        np.testing.assert_allclose(basis.T @ complement, 0.0, atol=1e-12)
        np.testing.assert_allclose(complement.T @ complement, np.eye(3), atol=1e-12)

    def test_random_subspace_stays_inside_container(self):
        container = np.eye(6)[:, :4]
        draw = random_subspace_within(container, 2, np.random.default_rng(3))

        self.assertEqual(draw.shape, (6, 2))
        np.testing.assert_allclose(draw.T @ draw, np.eye(2), atol=1e-12)
        projection_residual = draw - container @ (container.T @ draw)
        np.testing.assert_allclose(projection_residual, 0.0, atol=1e-12)

    def test_readout_basis_has_expected_rank(self):
        weights = np.column_stack([np.arange(5.0), 2 * np.arange(5.0)])
        self.assertEqual(readout_basis(weights).shape, (5, 1))

    def test_map_gap_decomposes_into_ceiling_and_transfer_penalty(self):
        import pandas as pd

        frame = pd.DataFrame({
            "fwd_cross_map": [0.4],
            "rev_cross_map": [0.6],
            "own_r1_map": [0.7],
            "own_r2_map": [0.5],
            "fwd_map_loss": [0.1],
            "rev_map_loss": [0.1],
        })
        result = add_map_decomposition(frame).iloc[0]

        self.assertAlmostEqual(result.raw_directional_gap, 0.2)
        self.assertAlmostEqual(result.target_ceiling_gap, 0.2)
        self.assertAlmostEqual(result.transfer_penalty_asymmetry, 0.0)
        self.assertAlmostEqual(result.decomposition_error, 0.0)

    def test_private_rescue_is_compared_with_matched_random_ablation(self):
        import pandas as pd

        frame = pd.DataFrame({
            "fwd_full": [0.4], "rev_full": [0.6],
            "fwd_shared": [0.5], "rev_shared": [0.6],
            "fwd_random_shared_mean": [0.43], "rev_random_shared_mean": [0.61],
            "fwd_minus_r1_private": [0.48], "rev_minus_r1_private": [0.59],
            "fwd_random_ablate_a_mean": [0.44], "rev_random_ablate_a_mean": [0.59],
            "fwd_minus_r2_private": [0.42], "rev_minus_r2_private": [0.58],
            "fwd_random_ablate_b_mean": [0.41], "rev_random_ablate_b_mean": [0.59],
        })
        result = add_subspace_contrasts(frame).iloc[0]

        self.assertAlmostEqual(result.selective_remove_r1_private, 0.09)
        self.assertAlmostEqual(result.selective_random_ablate_r1, 0.05)
        self.assertAlmostEqual(result.r1_private_excess_over_random, 0.04)

    def test_vectorized_hierarchical_interval_preserves_constant_metric(self):
        import pandas as pd

        frame = pd.DataFrame({
            "r2_session": np.repeat(["a", "b", "c"], 4),
            "metric": np.full(12, 0.25),
        })
        lo, hi = hierarchical_interval(
            frame, "metric", np.random.default_rng(8)
        )

        self.assertAlmostEqual(lo, 0.25)
        self.assertAlmostEqual(hi, 0.25)


if __name__ == "__main__":
    unittest.main()
