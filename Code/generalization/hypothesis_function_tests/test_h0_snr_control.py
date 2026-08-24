"""Tests for the pure H0 SNR-control helper functions.

These tests deliberately avoid loading NWB files. They verify the noise-injection
math used by `analyses/h0_snr_control.py`, which is the core of the H0 variance-match
control.
"""
from __future__ import annotations

import unittest
import sys
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERALIZATION_DIR = REPO_ROOT / "Code" / "generalization"
WHY_DIR = GENERALIZATION_DIR / "analyses"
for path in (str(GENERALIZATION_DIR), str(WHY_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# h0_snr_control imports the full data pipeline at module import time. Stub the
# NWB-heavy modules so these tests can exercise the pure noise helpers in a plain
# Python environment.
matplotlib_stub = types.ModuleType("matplotlib")
pyplot_stub = types.ModuleType("matplotlib.pyplot")
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)

decoder_utils_stub = types.ModuleType("decoder_utils")
decoder_utils_stub.DATA_DIR = REPO_ROOT / "Data"
sys.modules.setdefault("decoder_utils", decoder_utils_stub)

dimension_sweep_stub = types.ModuleType("dimension_sweep")
dimension_sweep_stub.align_full = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("align_full is not used by these unit tests")
)
sys.modules.setdefault("dimension_sweep", dimension_sweep_stub)

big_sweep_stub = types.ModuleType("big_sweep_phase2_crossday")
big_sweep_stub.SESSIONS_R1 = []
big_sweep_stub.SESSIONS_R2 = []
big_sweep_stub.EXCLUDE_TRIALS = {}
big_sweep_stub.N_PHASE_BINS = 30
big_sweep_stub.SMOOTH_SIGMA_MS = 50
big_sweep_stub.UNIT_QUALITIES = ("good", "mua")
big_sweep_stub.TRIAL_RESULTS = ("S", "F")
big_sweep_stub.filter_trials = lambda X, Y, meta, exclude: (X, Y, meta)
big_sweep_stub.kalman_fit_predict = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("kalman_fit_predict is not used by these unit tests")
)
big_sweep_stub.m2_per_trial = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("m2_per_trial is not used by these unit tests")
)
sys.modules.setdefault("big_sweep_phase2_crossday", big_sweep_stub)

from Code.generalization.analyses import h0_snr_control as h0


class TestH0SnrControlNoiseInjection(unittest.TestCase):
    def test_inject_to_match_raises_train_std_only_when_test_is_larger(self):
        rng = np.random.default_rng(123)
        n = 200_000
        y_train = np.column_stack([
            np.zeros(n),
            np.linspace(-1.0, 1.0, n),
            np.linspace(2.0, 4.0, n),
        ])
        y_test = np.column_stack([
            np.linspace(-3.0, 3.0, n),
            np.linspace(-0.25, 0.25, n),
            np.linspace(-4.0, 4.0, n),
        ])

        out = h0.inject_to_match(y_train, y_test, rng, dims=2)

        self.assertEqual(out.shape, y_train.shape)
        self.assertTrue(np.allclose(out[:, 2], y_train[:, 2]))
        self.assertGreater(out[:, 0].std(), y_train[:, 0].std())
        self.assertAlmostEqual(out[:, 0].std(), y_test[:, 0].std(), delta=0.01)
        self.assertAlmostEqual(out[:, 1].std(), y_train[:, 1].std(), delta=1e-12)

    def test_inject_to_match_does_not_mutate_input(self):
        rng = np.random.default_rng(1)
        y_train = np.ones((1000, 3))
        y_test = np.column_stack([
            np.linspace(-3.0, 3.0, 1000),
            np.linspace(-2.0, 2.0, 1000),
            np.linspace(-1.0, 1.0, 1000),
        ])
        original = y_train.copy()

        _ = h0.inject_to_match(y_train, y_test, rng, dims=2)

        np.testing.assert_array_equal(y_train, original)

    def test_inject_alpha_is_deterministic_for_same_seed(self):
        y_train = np.column_stack([
            np.linspace(-1.0, 1.0, 5000),
            np.linspace(0.0, 2.0, 5000),
            np.linspace(10.0, 11.0, 5000),
        ])

        out_a = h0.inject_alpha(y_train, 0.5, np.random.default_rng(7), dims=2)
        out_b = h0.inject_alpha(y_train, 0.5, np.random.default_rng(7), dims=2)

        np.testing.assert_allclose(out_a, out_b)
        self.assertTrue(np.allclose(out_a[:, 2], y_train[:, 2]))

    def test_inject_alpha_zero_returns_unchanged_values(self):
        y_train = np.arange(30.0).reshape(10, 3)

        out = h0.inject_alpha(y_train, 0.0, np.random.default_rng(7), dims=2)

        np.testing.assert_array_equal(out, y_train)


if __name__ == "__main__":
    unittest.main()
