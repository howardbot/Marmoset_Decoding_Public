from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # Code/ for local imports
from decoder_utils import (
    LAG_BINS, TARGET_MODES, UNIT_QUALITY_SETS, build_decoder_dataset, fit_ridge,
    load_nwb_and_reach,
    make_history_features, predict_linear, print_best_summary, print_top_results,
    quality_label, split_by_trial, standardize_train_test, state_names,
    summarize_train_test,
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

            for n_components in [5, 10, 20, 50, 100]:
                n_comp = min(n_components, F_tr.shape[0] - 1, F_tr.shape[1])
                pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0)
                Z_tr = pca.fit_transform(F_tr)
                Z_te = pca.transform(F_te)

                for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
                    coef = fit_ridge(Z_tr, Xt_tr, alpha)
                    row = {
                        "target_mode": target_mode,
                        "unit_set": unit_set,
                        "lag_bins": lag_bins,
                        "lag_ms": lag_bins * 50,
                        "history_bins": history_bins,
                        "history_ms": history_bins * 50,
                        "n_components": n_comp,
                        "alpha": alpha,
                        "explained_var": float(np.sum(pca.explained_variance_ratio_)),
                    }
                    row.update(summarize_train_test(
                        Xt_tr, predict_linear(Z_tr, coef),
                        Xt_te, predict_linear(Z_te, coef),
                        names,
                    ))
                    rows.append(row)

    result = pd.DataFrame(rows).sort_values("mean_test_r2", ascending=False)
    print("\n" + "#" * 78)
    print(f"PCA + Ridge: {target_mode} | units={unit_set}")
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
