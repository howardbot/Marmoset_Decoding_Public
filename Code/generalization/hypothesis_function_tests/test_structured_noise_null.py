"""Tests for structured residual-covariance injection."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GENERALIZATION = REPO / "Code" / "generalization"
WHY = GENERALIZATION / "analyses"
for path in (str(GENERALIZATION), str(WHY)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Stub NWB-heavy imports; these tests use only pure covariance helpers.
dimension_stub = types.ModuleType("dimension_sweep")
dimension_stub.align_full = None
sys.modules.setdefault("dimension_sweep", dimension_stub)
bridge_stub = types.ModuleType("global_state_bridge")
bridge_stub.kin_residual = None
bridge_stub.load = None
sys.modules.setdefault("global_state_bridge", bridge_stub)
big_stub = types.ModuleType("big_sweep_phase2_crossday")
big_stub.ANIMAL_SESSIONS = {}
big_stub.EXCLUDE_TRIALS = {}
big_stub.SESSIONS_R1 = []
big_stub.SESSIONS_R2 = []
big_stub.N_PHASE_BINS = 30
big_stub.SMOOTH_SIGMA_MS = 50
big_stub.UNIT_QUALITIES = ("good", "mua")
big_stub.TRIAL_RESULTS = ("S", "F")
big_stub.filter_trials = lambda X, Y, meta, exclude: (X, Y, meta)
big_stub.kalman_fit_predict = None
big_stub.m2_per_trial = None
sys.modules.setdefault("big_sweep_phase2_crossday", big_stub)

from Code.generalization.analyses.structured_noise_null import (
    inject_structured_increment,
    positive_covariance_increment,
)


class TestStructuredNoiseNull(unittest.TestCase):
    def test_increment_keeps_only_positive_covariance_difference(self):
        rng = np.random.default_rng(2)
        train = rng.normal(size=(200_000, 2)) @ np.diag([1.0, 3.0])
        target = rng.normal(size=(200_000, 2)) @ np.diag([2.0, 2.0])
        increment = positive_covariance_increment(train, target)

        eigenvalues = np.linalg.eigvalsh(increment)
        self.assertGreater(eigenvalues[-1], 2.5)
        self.assertAlmostEqual(eigenvalues[0], 0.0, delta=0.05)

    def test_injected_covariance_matches_positive_direction(self):
        rng = np.random.default_rng(5)
        train = rng.normal(size=(150_000, 2))
        target = rng.normal(size=(150_000, 2)) @ np.array([[2.0, 0.8], [0.0, 1.5]])
        original = train.copy()
        injected, increment = inject_structured_increment(
            train, train, target, np.random.default_rng(6)
        )

        np.testing.assert_array_equal(train, original)
        observed_increment = np.cov(injected, rowvar=False) - np.cov(train, rowvar=False)
        np.testing.assert_allclose(observed_increment, increment, atol=0.05)


if __name__ == "__main__":
    unittest.main()
