"""Lightweight tests for variability-matched cross-day decoder helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "analyses"))

from decode_variability_matched_crossday import (  # noqa: E402
    baseline_matrices,
    selected_trials_for_pair,
    subset_cache,
)


def test_selected_trials_for_pair_filters_all_keys():
    rows = pd.DataFrame(
        {
            "pair_id": [1, 1, 1, 2],
            "match_mode": ["joint", "joint", "neural", "joint"],
            "session": ["a", "a", "a", "a"],
            "trial": [3, 4, 5, 6],
            "selected": [True, False, True, True],
        }
    )
    np.testing.assert_array_equal(selected_trials_for_pair(rows, 1, "a"), [3])
    np.testing.assert_array_equal(
        selected_trials_for_pair(rows, 1, "a", match_mode="neural"), [5]
    )


def test_subset_cache_keeps_whole_requested_trials():
    raw = {
        "X": np.arange(24, dtype=float).reshape(8, 3),
        "neural": np.arange(32, dtype=float).reshape(8, 4),
        "meta": pd.DataFrame({"trial_number": [1, 1, 1, 1, 2, 2, 2, 2]}),
    }

    def fake_pca(neural, k):
        return neural[:, :2], np.zeros((neural.shape[1], 2)), neural.mean(0)

    def fake_average(neural_pc, meta, n_phase_bins):
        return np.zeros((n_phase_bins, neural_pc.shape[1]))

    with patch(
        "decode_variability_matched_crossday.pca_neural", fake_pca
    ), patch(
        "decode_variability_matched_crossday.trial_average_pc", fake_average
    ):
        cache = subset_cache(raw, [2])
    assert cache["n_trials"] == 1
    assert len(cache["X"]) == 4


def test_missing_sweep_requests_direct_baseline_recompute():
    with patch(
        "decode_variability_matched_crossday.SWEEP_CSV",
        Path("definitely_missing_phase2_sweep.csv"),
    ):
        assert baseline_matrices("relative_position") is None
