from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


WHY_DIR = Path(__file__).resolve().parents[1] / "analyses"
if str(WHY_DIR) not in sys.path:
    sys.path.insert(0, str(WHY_DIR))

import summarize_affine_centering_crossanimal as module


def synthetic_rows() -> pd.DataFrame:
    rows = []
    scores = {
        "original_concatenated": (0.28, 0.52),
        "original_trial_aware": (0.29, 0.51),
        "source_b": (0.30, 0.50),
        "behaviour_center": (0.39, 0.49),
        "target_b": (0.42, 0.48),
    }
    for condition, (forward, reverse) in scores.items():
        for direction, score in (("forward", forward), ("reverse", reverse)):
            rows.append({
                "animal": "TS",
                "q_context": "source",
                "condition": condition,
                "r1_session": "r1",
                "r2_session": "r2",
                "repeat": 0,
                "fold": 0,
                "direction": direction,
                "score": score,
            })
    return pd.DataFrame(rows)


def test_prepare_splits_uses_source_offset_as_unique_baseline():
    result = module.prepare_splits(synthetic_rows()).set_index("condition")
    assert np.isclose(result.at["source_b", "baseline_gap"], 0.20)
    assert np.isclose(result.at["source_b", "original_gap"], 0.24)
    assert np.isclose(result.at["source_b", "gap_closed_from_original"], 0.04)
    assert np.isclose(result.at["behaviour_center", "directional_gap"], 0.10)
    assert np.isclose(result.at["behaviour_center", "gap_closed"], 0.10)
    assert np.isclose(result.at["behaviour_center", "gap_closed_from_original"], 0.14)
    assert np.isclose(result.at["target_b", "gap_closed"], 0.14)


def test_prepare_splits_rejects_duplicate_directions():
    frame = synthetic_rows()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    try:
        module.prepare_splits(duplicate)
    except AssertionError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate direction row was accepted")
