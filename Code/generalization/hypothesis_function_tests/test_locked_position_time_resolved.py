from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WHY = Path(__file__).resolve().parents[1] / "analyses"
sys.path.insert(0, str(WHY))

from locked_position_time_resolved import (  # noqa: E402
    summarize_directional_gaps,
    time_resolved_metrics,
)


def test_time_resolved_correlation_is_across_trials():
    actual = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
            [5.0, 10.0, 15.0],
        ]
    )
    predicted = actual * 2.0
    meta = pd.DataFrame(
        {
            "trial_number": [1, 1, 2, 2, 3, 3],
            "local_bin": [0, 1, 0, 1, 0, 1],
        }
    )
    result = time_resolved_metrics(actual, predicted, meta, min_trials=3)
    np.testing.assert_allclose(result["corr_mean"], [1.0, 1.0])
    np.testing.assert_array_equal(result["n_trials"], [3, 3])


def test_time_resolved_metrics_reject_shape_mismatch():
    meta = pd.DataFrame({"trial_number": [1], "local_bin": [0]})
    try:
        time_resolved_metrics(np.zeros((1, 3)), np.zeros((2, 3)), meta)
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("shape mismatch should raise ValueError")


def test_directional_gap_summary_compares_paired_directions():
    rows = []
    for r1, offset in (("r1a", 0.00), ("r1b", 0.02), ("r1c", -0.01)):
        for time_bin in (1, 2, 12, 13):
            late = time_bin >= 12
            for direction, value in (
                ("R1->R2", 0.4 + offset),
                ("R2->R1", 0.4 + offset + (0.2 if late else 0.0)),
            ):
                rows.append(
                    {
                        "r2_session": "r2",
                        "r1_session": r1,
                        "direction": direction,
                        "time_bin": time_bin,
                        "corr_mean": value,
                    }
                )
    summary = summarize_directional_gaps(pd.DataFrame(rows))
    late = summary.loc[
        (summary["analysis"] == "directional_gap")
        & (summary["window"] == "late_360_510ms")
    ].iloc[0]
    np.testing.assert_allclose(late["mean_gap"], 0.2)
    assert late["positive_fraction"] == 1.0
