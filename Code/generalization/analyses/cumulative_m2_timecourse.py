"""Cumulative within-trial correlation matching the headline M2 metric.

The headline cross-day score averages, across trials and coordinates, the
temporal correlation between actual and predicted kinematics within each test
trial.  A single-bin across-trial correlation is therefore a different
estimand and cannot decompose that score.  This module instead recomputes the
same within-trial temporal correlation on progressively longer prefixes.  At
the final cutoff its value is numerically identical to ``m2_per_trial``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _prefix_correlation(
    actual: np.ndarray,
    predicted: np.ndarray,
    local_bins: np.ndarray,
    max_bin: int,
    min_rows: int = 4,
) -> np.ndarray:
    """Return prefix Pearson correlation, carried forward after trial end."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    local_bins = np.asarray(local_bins, dtype=int)
    order = np.argsort(local_bins)
    actual = actual[order]
    predicted = predicted[order]
    local_bins = local_bins[order]

    finite = np.isfinite(actual) & np.isfinite(predicted)
    x = np.where(finite, actual, 0.0)
    y = np.where(finite, predicted, 0.0)
    n = np.cumsum(finite.astype(float))
    sx = np.cumsum(x)
    sy = np.cumsum(y)
    sxx = np.cumsum(x * x)
    syy = np.cumsum(y * y)
    sxy = np.cumsum(x * y)
    numerator = n * sxy - sx * sy
    denominator = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    row_count = np.arange(1, len(actual) + 1)
    valid = (row_count >= min_rows) & (n >= 2) & (denominator > 1e-12)
    correlations = np.full(len(actual), np.nan, dtype=float)
    correlations[valid] = numerator[valid] / denominator[valid]

    result = np.full(max_bin + 1, np.nan, dtype=float)
    usable = local_bins <= max_bin
    result[local_bins[usable]] = correlations[usable]
    known = np.flatnonzero(np.isfinite(result))
    if len(known):
        last_known = np.maximum.accumulate(
            np.where(np.isfinite(result), np.arange(max_bin + 1), -1)
        )
        carry = last_known >= 0
        result[carry] = result[last_known[carry]]
    return result


def cumulative_m2_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    meta: pd.DataFrame,
    max_bin: int,
    bin_size_ms: int = 30,
) -> pd.DataFrame:
    """Evaluate headline-style within-trial correlation on every time prefix."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or len(actual) != len(meta):
        raise ValueError("actual, predicted, and meta must have matching shapes")
    if "local_bin" not in meta or "trial_number" not in meta:
        raise ValueError("meta must contain local_bin and trial_number")

    trajectories = []
    for _, indices in meta.groupby("trial_number", sort=False).indices.items():
        indices = np.asarray(indices)
        if len(indices) < 4:
            continue
        local_bins = meta.iloc[indices]["local_bin"].to_numpy(dtype=int)
        for dimension in range(actual.shape[1]):
            trajectories.append(
                _prefix_correlation(
                    actual[indices, dimension],
                    predicted[indices, dimension],
                    local_bins,
                    max_bin,
                )
            )
    if not trajectories:
        raise ValueError("no trials have at least four samples")
    values = np.stack(trajectories)
    with np.errstate(invalid="ignore"):
        cumulative = np.nanmean(values, axis=0)
    n_trial_dimensions = np.sum(np.isfinite(values), axis=0)
    first_valid = int(np.flatnonzero(np.isfinite(cumulative))[0])
    bins = np.arange(first_valid, max_bin + 1)
    return pd.DataFrame(
        {
            "time_bin": bins,
            "time_end_ms": (bins + 1) * bin_size_ms,
            "cumulative_corr": cumulative[bins],
            "n_trial_dimensions": n_trial_dimensions[bins],
        }
    )
