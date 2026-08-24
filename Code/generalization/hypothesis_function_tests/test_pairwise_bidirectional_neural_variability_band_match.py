"""Tests for matching pair-MSD means to a fixed day's mean +/- SD band."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "analyses"))

from pairwise_bidirectional_neural_variability_band_match import (  # noqa: E402
    pair_distribution_stats,
    trim_pair_mean_to_band,
)


def _distance_matrix(points):
    points = np.asarray(points, dtype=float)
    return (points[:, None] - points[None, :]) ** 2


def test_pair_distribution_stats_use_unique_pair_values():
    matrix = _distance_matrix([0.0, 1.0, 3.0])
    values = np.array([1.0, 9.0, 4.0])
    stats = pair_distribution_stats(matrix, np.arange(3))
    np.testing.assert_allclose(stats["mean"], values.mean())
    np.testing.assert_allclose(stats["sd"], values.std(ddof=1))
    assert stats["n_pairs"] == 3


def test_pair_mean_down_trim_enters_anchor_band():
    result = trim_pair_mean_to_band(
        _distance_matrix([0.0, 1.0, 2.0, 10.0]), 1.5, 2.5
    )
    assert result["direction"] == "down"
    assert result["within_band"]
    assert set(result["keep"]) == {0, 1, 2}


def test_pair_mean_up_trim_enters_anchor_band():
    result = trim_pair_mean_to_band(
        _distance_matrix([-2.0, -1.0, 0.0, 1.0, 2.0]), 6.0, 7.0
    )
    assert result["direction"] == "up"
    assert result["within_band"]
    assert set(result["keep"]) == {0, 1, 3, 4}


def test_pair_mean_inside_band_keeps_all_trials():
    matrix = _distance_matrix([0.0, 1.0, 2.0, 3.0])
    start = pair_distribution_stats(matrix, np.arange(4))["mean"]
    result = trim_pair_mean_to_band(matrix, start - 0.1, start + 0.1)
    assert result["direction"] == "inside"
    assert result["within_band"]
    assert len(result["keep"]) == 4
