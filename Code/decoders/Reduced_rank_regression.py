from __future__ import annotations

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # Code/ for local imports
from decoder_utils import (
    LAG_BINS, TARGET_MODES, UNIT_QUALITY_SETS, build_decoder_dataset, fit_ridge,
    load_nwb_and_reach,
    make_history_features, predict_linear, print_best_summary, print_top_results,
    quality_label, split_by_trial, standardize_train_test, state_names,
    summarize_train_test,
)


def reduce_rank(train_pred, test_pred, rank):
    mean = train_pred.mean(axis=0)
    centered = train_pred - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank].T
    return (
        mean + (train_pred - mean) @ basis @ basis.T,
        mean + (test_pred - mean) @ basis @ basis.T,
    )


def run_target(nwb_prc, reach_tbl, target_mode, unit_qualities):
    names = state_names(target_mode)
    unit_set = quality_label(unit_qualities)
    X, Y, meta = build_decoder_dataset(
        nwb_prc,
        reach_tbl,
        target_mode,
        unit_qualities=unit_qualities,
        trial_window="start_to_peak",
    )
    X_tr, Y_tr, X_te, Y_te, meta_tr, meta_te = split_by_trial(X, Y, meta)

    rows = []
    for history_bins in [1, 2, 5, 10]:
        for lag_bins in LAG_BINS:
            F_tr, Xt_tr, _ = make_history_features(
                X_tr, Y_tr, meta_tr, history_bins, lag_bins=lag_bins
            )
            F_te, Xt_te, _ = make_history_features(
                X_te, Y_te, meta_te, history_bins, lag_bins=lag_bins
            )
            if len(F_tr) == 0 or len(F_te) == 0:
                continue
            F_tr, F_te = standardize_train_test(F_tr, F_te)

            for alpha in [10.0, 100.0, 1000.0, 10000.0]:
                coef = fit_ridge(F_tr, Xt_tr, alpha)
                pred_tr = predict_linear(F_tr, coef)
                pred_te = predict_linear(F_te, coef)

                for rank in range(1, min(Xt_tr.shape[1], 6) + 1):
                    rr_tr, rr_te = reduce_rank(pred_tr, pred_te, rank)
                    row = {
                        "target_mode": target_mode,
                        "unit_set": unit_set,
                        "lag_bins": lag_bins,
                        "lag_ms": lag_bins * 50,
                        "history_bins": history_bins,
                        "history_ms": history_bins * 50,
                        "alpha": alpha,
                        "rank": rank,
                    }
                    row.update(summarize_train_test(Xt_tr, rr_tr, Xt_te, rr_te, names))
                    rows.append(row)

    result = pd.DataFrame(rows).sort_values("mean_test_r2", ascending=False)
    print("\n" + "#" * 78)
    print(f"Reduced-rank regression: {target_mode} | units={unit_set}")
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
