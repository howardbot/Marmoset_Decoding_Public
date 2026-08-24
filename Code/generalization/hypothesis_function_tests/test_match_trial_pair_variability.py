"""Unit tests for equal-N trial-pair variability matching helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "analyses"))

from match_trial_pair_variability import (  # noqa: E402
    choose_largest_match,
    greedy_path,
    pair_matrix,
    subset_mean,
)


def test_pair_matrix_roundtrip():
    pairs = pd.DataFrame(
        {
            "trial_i": [1, 1, 2],
            "trial_j": [2, 3, 3],
            "value": [2.0, 4.0, 6.0],
        }
    )
    ids, matrix = pair_matrix(pairs, "value")
    np.testing.assert_array_equal(ids, [1, 2, 3])
    np.testing.assert_allclose(
        matrix,
        [[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]],
    )


def test_greedy_paths_move_means_in_requested_directions():
    matrix = np.array(
        [
            [0.0, 1.0, 1.0, 8.0],
            [1.0, 0.0, 1.0, 8.0],
            [1.0, 1.0, 0.0, 8.0],
            [8.0, 8.0, 8.0, 0.0],
        ]
    )
    lower = greedy_path(matrix, min_trials=3, direction="lower")
    raise_ = greedy_path(matrix, min_trials=3, direction="raise")
    assert subset_mean(matrix, lower[3]) < subset_mean(matrix, lower[4])
    assert subset_mean(matrix, raise_[3]) > subset_mean(matrix, raise_[4])


def test_choose_largest_match_prefers_equal_n_within_tolerance():
    r1 = np.array(
        [
            [0.0, 2.0, 2.0, 8.0],
            [2.0, 0.0, 2.0, 8.0],
            [2.0, 2.0, 0.0, 8.0],
            [8.0, 8.0, 8.0, 0.0],
        ]
    )
    r2 = np.array(
        [
            [0.0, 4.0, 4.0, 10.0],
            [4.0, 0.0, 4.0, 10.0],
            [4.0, 4.0, 0.0, 10.0],
            [10.0, 10.0, 10.0, 0.0],
        ]
    )
    result = choose_largest_match(r1, r2, min_trials=3, tolerance=0.5)
    assert len(result["keep_r1"]) == len(result["keep_r2"])
    assert result["n_trials"] >= 3
