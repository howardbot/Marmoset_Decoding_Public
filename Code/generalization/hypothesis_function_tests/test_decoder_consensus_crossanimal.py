"""Unit tests for the cross-animal decoder-consensus helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

WHY = Path(__file__).resolve().parents[1] / "analyses"
GENERALIZATION = WHY.parent
CODE = GENERALIZATION.parent
for path in (WHY, GENERALIZATION, CODE):
    sys.path.insert(0, str(path))

from decoder_consensus_crossanimal import fit_arx_trial_aware
from common_target_source_test import select_common_target
from decoder_model_audit import RIDGE_ALPHA, _fit_linear
from shared_mapping_stability_crossanimal import (
    cosine_similarity,
    split_calibration_trials,
)
from summarize_decoder_consensus_crossanimal import exact_label_permutation


def test_trial_aware_arx_matches_only_within_trial_transitions():
    state = np.asarray([[0.0], [1.0], [2.0], [100.0], [101.0], [102.0]])
    activity = np.asarray([[0.0], [0.2], [0.4], [3.0], [3.2], [3.4]])
    meta = pd.DataFrame({"trial_number": [1, 1, 1, 2, 2, 2]})
    calibration = np.ones(len(meta), dtype=bool)
    observed = fit_arx_trial_aware(
        activity, state, calibration, meta
    )
    source = np.asarray([0, 1, 3, 4])
    target = np.asarray([1, 2, 4, 5])
    features = np.column_stack([state[source], activity[target]])
    expected = _fit_linear(features, state[target], RIDGE_ALPHA)
    np.testing.assert_allclose(observed, expected)


def test_split_calibration_trials_is_disjoint_whole_trial_partition():
    meta = pd.DataFrame({
        "trial_number": np.repeat(np.arange(8), 3),
    })
    calibration = meta.trial_number.isin([0, 1, 2, 3, 4, 5]).to_numpy()
    first, second = split_calibration_trials(meta, calibration, seed=11)
    assert not np.any(first & second)
    np.testing.assert_array_equal(first | second, calibration)
    for trial, indices in meta.groupby("trial_number").indices.items():
        indices = np.asarray(indices)
        if trial < 6:
            assert first[indices].all() or second[indices].all()
        else:
            assert not (first[indices].any() or second[indices].any())


def test_cosine_similarity_handles_scale_and_zero_norm():
    assert np.isclose(cosine_similarity([1, 2], [2, 4]), 1.0)
    assert np.isclose(cosine_similarity([1, 0], [-1, 0]), -1.0)
    assert np.isnan(cosine_similarity([0, 0], [1, 0]))


def test_exact_label_permutation_enumerates_all_assignments():
    values = np.asarray([0.0, 1.0, 2.0, 3.0])
    labels = np.asarray(["R1", "R1", "R1", "R2"])
    difference, p_value = exact_label_permutation(values, labels)
    assert np.isclose(difference, 2.0)
    assert np.isclose(p_value, 0.5)


def test_common_target_never_returns_excluded_source():
    sessions = ("a", "b", "c")
    observed = {
        select_common_target(sessions, "b", index)
        for index in range(8)
    }
    assert observed == {"a", "c"}
