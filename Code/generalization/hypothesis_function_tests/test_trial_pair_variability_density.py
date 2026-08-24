"""Unit tests for per-day trial-pair variability helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "diagnostics"))

from trial_pair_variability_density import (  # noqa: E402
    pairwise_mean_squared_difference,
    phase_resample_joint,
)


def test_pairwise_mean_squared_difference_known_values():
    stack = np.array(
        [
            [[0.0], [0.0]],
            [[1.0], [1.0]],
            [[3.0], [3.0]],
        ]
    )
    # scipy condensed pair order is (0,1), (0,2), (1,2).
    np.testing.assert_allclose(
        pairwise_mean_squared_difference(stack),
        [1.0, 9.0, 4.0],
    )


def test_pairwise_metric_averages_over_features():
    one_feature = np.array([[[0.0]], [[2.0]]])
    duplicated_feature = np.repeat(one_feature, 5, axis=2)
    np.testing.assert_allclose(
        pairwise_mean_squared_difference(one_feature),
        pairwise_mean_squared_difference(duplicated_feature),
    )


def test_phase_resample_joint_preserves_trial_alignment_and_endpoints():
    position = np.array([[0.0], [1.0], [2.0], [10.0], [12.0], [14.0], [16.0]])
    neural = position * 3.0
    meta = pd.DataFrame({"trial_number": [7, 7, 7, 9, 9, 9, 9]})

    trial_ids, position_trials, neural_trials = phase_resample_joint(
        position, neural, meta, n_phase=5
    )

    np.testing.assert_array_equal(trial_ids, [7, 9])
    np.testing.assert_allclose(position_trials[:, 0, 0], [0.0, 10.0])
    np.testing.assert_allclose(position_trials[:, -1, 0], [2.0, 16.0])
    np.testing.assert_allclose(neural_trials, position_trials * 3.0)
