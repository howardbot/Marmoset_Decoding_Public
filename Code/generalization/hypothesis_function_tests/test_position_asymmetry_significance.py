import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WHY_DIR = Path(__file__).resolve().parents[1] / "analyses"
sys.path.insert(0, str(WHY_DIR))

from position_asymmetry_significance import (  # noqa: E402
    build_sweep_pair_gaps,
    fdr_bh,
    gap_test,
)


def test_fdr_bh_preserves_order_and_nan():
    adjusted = fdr_bh(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_gap_test_is_paired_t_on_directional_differences():
    forward = np.array([0.1, 0.2, 0.3, 0.4])
    reverse = np.array([0.3, 0.35, 0.55, 0.7])
    expected = stats.ttest_rel(reverse, forward, alternative="greater")
    observed = gap_test(reverse - forward)
    assert np.isclose(observed["t_statistic"], expected.statistic)
    assert np.isclose(observed["p_one_sided_gt0"], expected.pvalue)
    assert observed["mean_gap"] > 0


def test_sweep_pairing_prefers_excluded_outlier_row():
    r1 = "r1"
    r2 = "r2"
    common = {
        "bin_size_ms": 30,
        "smoother": "butter_o2",
        "lag_ms": 0,
        "decoder": "kalman",
        "target_mode": "relative_position",
        "history_ms": np.nan,
    }
    frame = pd.DataFrame(
        [
            {
                **common,
                "train_session": r1,
                "test_session": r2,
                "outlier_mode": "include",
                "M2_mean": 0.2,
            },
            {
                **common,
                "train_session": r1,
                "test_session": r2,
                "outlier_mode": "exclude",
                "M2_mean": 0.4,
            },
            {
                **common,
                "train_session": r2,
                "test_session": r1,
                "outlier_mode": "include",
                "M2_mean": 0.7,
            },
            {
                **common,
                "train_session": r2,
                "test_session": r1,
                "outlier_mode": "exclude",
                "M2_mean": 0.8,
            },
        ]
    )
    paired = build_sweep_pair_gaps(frame, (r1,), (r2,))
    assert len(paired) == 1
    assert np.isclose(paired.iloc[0]["R1->R2"], 0.4)
    assert np.isclose(paired.iloc[0]["R2->R1"], 0.8)
    assert np.isclose(paired.iloc[0]["gap"], 0.4)
