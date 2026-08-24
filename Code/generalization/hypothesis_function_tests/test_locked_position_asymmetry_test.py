import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WHY_DIR = Path(__file__).resolve().parents[1] / "analyses"
sys.path.insert(0, str(WHY_DIR))

from locked_position_asymmetry_test import (  # noqa: E402
    one_sample_test,
    paired_directions,
)


def test_pairing_preserves_exact_r1_r2_pair():
    frame = pd.DataFrame(
        {
            "train_session": ["r1a", "R2_interferenceAndInterferenceFree"],
            "test_session": ["R2_interferenceAndInterferenceFree", "r1a"],
            "correlation": [0.2, 0.5],
        }
    )
    paired = paired_directions(frame)
    assert len(paired) == 1
    assert np.isclose(paired.iloc[0]["gap"], 0.3)


def test_one_sample_test_matches_directional_t_test():
    values = np.array([0.1, 0.2, 0.3, -0.05])
    expected = stats.ttest_1samp(values, 0.0, alternative="greater")
    observed = one_sample_test(values)
    assert np.isclose(observed["t"], expected.statistic)
    assert np.isclose(observed["p_one_sided_gt0"], expected.pvalue)
