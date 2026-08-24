"""Tests for paired session-cluster summaries."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Code.generalization.analyses.session_clustered_asymmetry import (
    hierarchical_bootstrap,
    pair_directions,
)


class TestSessionClusteredAsymmetry(unittest.TestCase):
    def test_pair_directions_matches_same_sessions(self):
        rows = []
        for r1 in ("r1a", "r1b"):
            for r2 in ("r2a", "r2b"):
                rows.append({"pair_category": "R1->R2", "train_session": r1,
                             "test_session": r2, "corr": 0.2})
                rows.append({"pair_category": "R2->R1", "train_session": r2,
                             "test_session": r1, "corr": 0.5})
        paired = pair_directions(pd.DataFrame(rows))

        self.assertEqual(len(paired), 4)
        np.testing.assert_allclose(paired.asymmetry, 0.3)

    def test_hierarchical_bootstrap_preserves_constant_contrast(self):
        paired = pd.DataFrame({
            "r2_session": np.repeat(["r2a", "r2b", "r2c"], 4),
            "asymmetry": np.full(12, 0.25),
        })
        boot = hierarchical_bootstrap(paired, np.random.default_rng(3), n_boot=100)

        np.testing.assert_allclose(boot, 0.25)


if __name__ == "__main__":
    unittest.main()
