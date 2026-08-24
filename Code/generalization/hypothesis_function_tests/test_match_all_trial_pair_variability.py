"""Unit tests for all-pair neural/position variability matching."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "analyses"))

from match_all_trial_pair_variability import (  # noqa: E402
    _means_after_each_removal,
    choose_joint_match,
    choose_single_metric_match,
    date_label,
)


def _distance_matrix(points):
    points = np.asarray(points, dtype=float)
    return (points[:, None] - points[None, :]) ** 2


def test_date_label_supports_both_animals():
    assert date_label("TSAL20250801_0830_staticAndStaticFree") == "20250801"
    assert date_label("TYTR20250206_0830_staticAndStaticFree001") == "20250206"


def test_means_after_each_removal_matches_direct_calculation():
    matrix = _distance_matrix([0.0, 1.0, 2.0, 6.0])
    keep = np.arange(4)
    calculated = _means_after_each_removal(matrix, keep)
    direct = []
    for remove in range(4):
        sub = np.delete(np.delete(matrix, remove, axis=0), remove, axis=1)
        direct.append(sub[np.triu_indices(3, k=1)].mean())
    np.testing.assert_allclose(calculated, direct)


def test_single_metric_match_adapts_to_reversed_group_direction():
    high = _distance_matrix([0.0, 2.0, 5.0, 9.0, 15.0])
    low = _distance_matrix([0.0, 1.0, 2.0, 3.0, 4.0])
    result = choose_single_metric_match(high, low, min_trials=3, tolerance=0.5)
    assert result["direction_r1"] == "adaptive"
    assert result["direction_r2"] == "adaptive"
    assert len(result["keep_r1"]) == len(result["keep_r2"])


def test_joint_match_returns_equal_n_and_both_gap_fields():
    neural1 = _distance_matrix([0.0, 1.0, 2.0, 3.0, 7.0])
    neural2 = _distance_matrix([0.0, 1.5, 3.0, 4.5, 8.0, 10.0])
    position1 = _distance_matrix([0.0, 2.0, 3.0, 5.0, 6.0])
    position2 = _distance_matrix([0.0, 1.0, 4.0, 5.0, 7.0, 9.0])
    result = choose_joint_match(
        neural1, position1, neural2, position2, min_trials=3, tolerance=0.5
    )
    assert len(result["keep_r1"]) == len(result["keep_r2"])
    assert "neural_gap" in result
    assert "position_gap" in result
