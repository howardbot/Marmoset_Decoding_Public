"""Tests for the cross-animal Kalman/Wiener comparison helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "analyses"))

from compare_kalman_wiener_crossanimal import summarize  # noqa: E402


def test_summarize_computes_directional_and_decoder_gap_differences():
    pairs = pd.DataFrame(
        {
            "animal": ["TS"] * 4,
            "target": ["relative_position"] * 4,
            "decoder": ["kalman", "kalman", "wiener", "wiener"],
            "r2_session": ["b", "b", "b", "b"],
            "forward_corr": [0.2, 0.4, 0.3, 0.5],
            "reverse_corr": [0.5, 0.5, 0.4, 0.5],
            "directional_gap": [0.3, 0.1, 0.1, 0.0],
        }
    )
    summary, by_day = summarize(pairs)
    kalman = summary[summary.decoder == "kalman"].iloc[0]
    wiener = summary[summary.decoder == "wiener"].iloc[0]
    assert np.isclose(kalman.directional_gap, 0.2)
    assert np.isclose(wiener.directional_gap, 0.05)
    assert np.isclose(kalman.wiener_minus_kalman_gap, -0.15)
    assert len(by_day) == 2
