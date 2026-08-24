"""Unit tests for one-sided R2 trimming into an R1 STD band."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "analyses"))

from trim_r2_to_r1_std_band import (  # noqa: E402
    r1_band,
    trajectory_std,
    trim_to_band,
    trim_to_target,
)


def _distance_matrix(points):
    points = np.asarray(points, dtype=float)
    return (points[:, None] - points[None, :]) ** 2


def test_trajectory_std_matches_sample_std_for_scalar_trajectories():
    points = np.array([-2.0, -1.0, 1.0, 2.0])
    matrix = _distance_matrix(points)
    calculated = trajectory_std(matrix, np.arange(len(points)))
    np.testing.assert_allclose(calculated, points.std(ddof=1))


def test_down_trim_removes_extreme_trial_and_enters_band():
    result = trim_to_band(_distance_matrix([0.0, 1.0, 2.0, 10.0]), 0.8, 1.2)
    assert result["direction"] == "down"
    assert result["within_band"]
    assert result["status"] == "entered_band"
    assert set(result["keep"]) == {0, 1, 2}


def test_up_trim_removes_central_trial_and_enters_band():
    result = trim_to_band(_distance_matrix([-2.0, -1.0, 0.0, 1.0, 2.0]), 1.7, 1.9)
    assert result["direction"] == "up"
    assert result["within_band"]
    assert result["status"] == "entered_band"
    assert set(result["keep"]) == {0, 1, 3, 4}


def test_inside_band_keeps_all_trials():
    result = trim_to_band(_distance_matrix([0.0, 1.0, 2.0, 3.0]), 1.0, 1.5)
    assert result["direction"] == "inside"
    assert result["status"] == "inside_initial"
    assert result["selected_step"] == 0
    assert len(result["keep"]) == 4


def test_r1_band_uses_sample_sd_across_days():
    values = np.array([1.0, 2.0, 3.0])
    band = r1_band(values)
    assert band == {
        "mean": 2.0,
        "across_day_sd": 1.0,
        "lower": 1.0,
        "upper": 3.0,
    }


def test_trim_to_target_selects_closest_side_of_first_crossing():
    result = trim_to_target(
        _distance_matrix([0.0, 1.0, 2.0, 10.0]),
        target_std=1.2,
    )
    assert result["direction"] == "down"
    assert result["status"] == "crossed_target"
    assert set(result["keep"]) == {0, 1, 2}
    np.testing.assert_allclose(result["selected_std"], 1.0)


def test_trim_to_target_can_raise_std_by_removing_central_trial():
    result = trim_to_target(
        _distance_matrix([-2.0, -1.0, 0.0, 1.0, 2.0]),
        target_std=1.75,
    )
    assert result["direction"] == "up"
    assert result["status"] == "crossed_target"
    assert set(result["keep"]) == {0, 1, 3, 4}
