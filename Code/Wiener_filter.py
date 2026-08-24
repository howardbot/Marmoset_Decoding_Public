"""Wiener Filter decoder using the official KordingLab Neural_Decoding implementation.

``WienerFilterRegression`` is a thin wrapper around
``sklearn.linear_model.LinearRegression`` (OLS, no regularization). History
embedding and lag are applied externally via ``make_history_features`` /
``apply_lag`` from ``decoder_utils`` so the neural input matrix matches the
Kording reference format.

History length is configured in **physical units (ms)** rather than bin count
so the same target stays comparable across bin sizes. ``history_bins`` is the
integer that minimises ``|history_bins * bin_size_ms - target_ms|`` (minimum
1), i.e. nearest rounding rather than floor. The realised ``history_ms`` is
recorded in the results table.

Mean offsets on neural features and behavioural targets are removed explicitly
on the training set and re-added on the test set, matching the centering used
in ``cross_day_decoder.kalman_fit_predict``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from Neural_Decoding.decoders import WienerFilterRegression

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # Code/ for local imports
from decoder_utils import (
    LAG_BINS, TARGET_MODES, UNIT_QUALITY_SETS, build_decoder_dataset,
    load_nwb_and_reach, make_history_features, print_best_summary,
    print_top_results, quality_label, split_by_trial, state_names,
    summarize_train_test,
)

# Sweep targets in milliseconds. Each value is rounded *down* to the nearest
# bin so it never exceeds the requested look-back.
HISTORY_MS_OPTIONS = [50, 100]

# Default decoder bin size in seconds. Kept in sync with ``build_decoder_dataset``
# so the ms -> bins conversion is consistent. Override via ``run_target(..., bin_size=...)``.
DEFAULT_BIN_SIZE_S = 0.01


def history_bins_for(target_ms: float, bin_size_ms: float) -> int:
    """Convert a physical history target (ms) into an integer bin count.

    Picks the integer that minimises the absolute distance to ``target_ms``
    (i.e. nearest rounding, not floor). Clamped to a minimum of 1 because
    zero bins is not a valid FIR length. Half-integer ratios round up
    (Python's ``round`` would use banker's rounding; we want explicit
    "ceil on tie" so the closer-of-equal-distance pick favours more history).
    """
    ratio = target_ms / bin_size_ms
    n = int(np.floor(ratio + 0.5))  # round half up
    return max(1, n)


def fit_predict_wiener(F_tr, X_tr, F_te):
    """Train Kording-lab Wiener filter with explicit mean centering and predict.

    The training-set means of both the neural feature matrix and the behavioural
    targets are subtracted before the OLS fit, then the target mean is added
    back to the test-set predictions. This makes the intercept handling
    explicit and parallel to ``kalman_fit_predict`` elsewhere in the codebase.
    """
    F_mean = np.nanmean(F_tr, axis=0)
    X_mean = np.nanmean(X_tr, axis=0)
    F_tr_c = F_tr - F_mean
    X_tr_c = X_tr - X_mean
    F_te_c = F_te - F_mean

    model = WienerFilterRegression()
    model.fit(F_tr_c, X_tr_c)
    pred = model.predict(F_te_c) + X_mean
    return pred, model, (F_mean, X_mean)


def run_target(nwb_prc, reach_tbl, target_mode, unit_qualities,
               bin_size=DEFAULT_BIN_SIZE_S,
               history_ms_options=HISTORY_MS_OPTIONS):
    names = state_names(target_mode)
    unit_set = quality_label(unit_qualities)
    bin_size_ms = bin_size * 1000.0

    X, Y, meta = build_decoder_dataset(
        nwb_prc,
        reach_tbl,
        target_mode,
        bin_size=bin_size,
        unit_qualities=unit_qualities,
        trial_window="start_to_peak",
    )
    X_tr, Y_tr, X_te, Y_te, meta_tr, meta_te = split_by_trial(X, Y, meta)

    rows = []
    for history_ms in history_ms_options:
        history_bins = history_bins_for(history_ms, bin_size_ms)
        realised_history_ms = history_bins * bin_size_ms
        for lag_bins in LAG_BINS:
            F_tr, Xt_tr, _ = make_history_features(
                X_tr, Y_tr, meta_tr, history_bins, lag_bins=lag_bins
            )
            F_te, Xt_te, _ = make_history_features(
                X_te, Y_te, meta_te, history_bins, lag_bins=lag_bins
            )
            if len(F_tr) == 0 or len(F_te) == 0:
                continue

            # Train with explicit mean centering on the train fold.
            F_mean = np.nanmean(F_tr, axis=0)
            X_mean = np.nanmean(Xt_tr, axis=0)
            model = WienerFilterRegression()
            model.fit(F_tr - F_mean, Xt_tr - X_mean)
            pred_tr = model.predict(F_tr - F_mean) + X_mean
            pred_te = model.predict(F_te - F_mean) + X_mean

            row = {
                "target_mode": target_mode,
                "unit_set": unit_set,
                "lag_bins": lag_bins,
                "lag_ms": lag_bins * bin_size_ms,
                "history_bins": history_bins,
                "history_ms_target": history_ms,
                "history_ms_realised": realised_history_ms,
                "bin_size_ms": bin_size_ms,
            }
            row.update(summarize_train_test(Xt_tr, pred_tr, Xt_te, pred_te, names))
            rows.append(row)

    result = pd.DataFrame(rows).sort_values("mean_test_r2", ascending=False)
    print("\n" + "#" * 78)
    print(f"Wiener filter (KordingLab): {target_mode} | units={unit_set} | bin={bin_size_ms:.0f}ms")
    print("#" * 78)
    print_top_results(result)
    return result.iloc[0].to_dict()


def main():
    io, nwb_prc, reach_tbl = load_nwb_and_reach()
    try:
        best = [
            run_target(nwb_prc, reach_tbl, target_mode, unit_qualities)
            for target_mode in TARGET_MODES
            for unit_qualities in UNIT_QUALITY_SETS
        ]
        print_best_summary(best)
    finally:
        io.close()


if __name__ == "__main__":
    main()
