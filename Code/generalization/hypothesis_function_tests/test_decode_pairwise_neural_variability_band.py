"""Tests for decoding pair-specific neural variability-band selections."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "analyses"))

from decode_pairwise_neural_variability_band import (  # noqa: E402
    selected_trials_for_cell,
    summarize_asymmetry,
)


def test_selected_trials_for_cell_filters_pair_direction_and_session():
    rows = pd.DataFrame(
        {
            "pair_id": [1, 1, 1, 2],
            "match_direction": ["trim_r2_to_r1", "trim_r2_to_r1", "trim_r1_to_r2", "trim_r2_to_r1"],
            "trimmed_session": ["r2", "r2", "r1", "r2"],
            "trial": [4, 7, 3, 9],
            "selected": [True, False, True, True],
        }
    )
    selected = selected_trials_for_cell(rows, 1, "trim_r2_to_r1", "r2")
    np.testing.assert_array_equal(selected, [4])


def test_summarize_asymmetry_uses_paired_directional_difference():
    rows = []
    for pair_id, forward, reverse in ((1, 0.2, 0.5), (2, 0.4, 0.6)):
        for direction, matched in (("R1->R2", forward), ("R2->R1", reverse)):
            rows.append(
                {
                    "pair_id": pair_id,
                    "target_mode": "relative_position",
                    "match_direction": "trim_r2_to_r1",
                    "decoder_direction": direction,
                    "baseline_corr": matched - 0.1,
                    "matched_corr": matched,
                }
            )
    summary = summarize_asymmetry(pd.DataFrame(rows)).iloc[0]
    np.testing.assert_allclose(summary["matched_signed_gap"], 0.25)
    np.testing.assert_allclose(summary["baseline_signed_gap"], 0.25)
    np.testing.assert_allclose(summary["mean_absolute_cell_gap"], 0.25)
