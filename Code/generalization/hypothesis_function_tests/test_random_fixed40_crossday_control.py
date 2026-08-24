"""Unit tests for the random fixed-N cross-day control."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "analyses"))

from random_fixed40_crossday_control import sample_trial_ids  # noqa: E402


def test_sample_trial_ids_is_unique_sorted_and_deterministic():
    first = sample_trial_ids(range(20), 8, np.random.default_rng(7))
    second = sample_trial_ids(range(20), 8, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, np.unique(first))
    assert len(first) == 8


def test_sample_trial_ids_rejects_oversampling():
    try:
        sample_trial_ids([1, 2], 3, np.random.default_rng(0))
    except ValueError:
        return
    raise AssertionError("oversampling should raise ValueError")
