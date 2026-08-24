"""Mahalanobis coverage controls under the locked cross-day decoder config.

This is the decoder-matched companion to ``coverage_controls.py``.  It uses
the exact locked preprocessing/alignment path from the big cross-day sweep:

    30-ms bins, Butterworth order 2, 0-ms neural lag, K_PCS=12,
    phase-average CCA, and either Kalman or Wiener (50-ms history).

For each ordered train/test pair it measures:

1. Per-bin decode error versus kinematic and decoder-input Mahalanobis distance
   from the training distribution.
2. Full-test versus in-support-test decode correlation, where in-support means
   below the training distribution's 95th-percentile kinematic distance.

The within-R1 error-distance relationship is the baseline for the cross-epoch
R1->R2 relationship.  Session-pair rows are descriptive robustness checks, not
independent biological replicates.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from Wiener_filter import history_bins_for
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS,
    EXCLUDE_TRIALS,
    K_PCS,
    build_cache_entry,
    kalman_fit_predict,
    m2_per_trial,
    wiener_fit_predict,
)
from manifold_align import apply_alignment, cca_align

warnings.filterwarnings("ignore")

BIN_MS = 30
SMOOTHER_KW = {
    "smoother": "butter",
    "smooth_cutoff_hz": 6.0,
    "smooth_order": 2,
}
WIENER_HISTORY_MS = 50
TARGETS = ("relative_position", "relative_velocity")
DECODERS = ("kalman", "wiener")
OUT = _THIS.parents[2] / "Results" / "manifold_geometry"
FIG = OUT / "figures"


def mahalanobis(points, mean, inverse_covariance):
    delta = points - mean
    squared = np.einsum(
        "ij,jk,ik->i", delta, inverse_covariance, delta, optimize=True
    )
    return np.sqrt(np.clip(squared, 0, None))


def distances_to_training(train, test, ridge):
    mean = np.nanmean(train, axis=0)
    covariance = np.cov(train.T)
    covariance = np.atleast_2d(covariance)
    inverse = np.linalg.pinv(
        covariance + ridge * np.eye(covariance.shape[0]),
        hermitian=True,
    )
    return (
        mahalanobis(train, mean, inverse),
        mahalanobis(test, mean, inverse),
    )


def safe_corr(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() <= 10:
        return np.nan
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


def zscore(values):
    scale = np.nanstd(values)
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / scale


def align_pair(train_cache, test_cache):
    w_train, w_test, mean_train, mean_test = cca_align(
        train_cache["traj"], test_cache["traj"]
    )
    neural_train = apply_alignment(
        train_cache["Y_pc"], w_train, mean_train
    )
    neural_test = apply_alignment(
        test_cache["Y_pc"], w_test, mean_test
    )
    return neural_train, neural_test


def decoder_arrays(train_cache, test_cache, neural_train, neural_test, decoder):
    if decoder == "kalman":
        true_test, prediction = kalman_fit_predict(
            train_cache["X"],
            neural_train,
            test_cache["X"],
            neural_test,
            test_cache["meta"],
        )
        return {
            "train_kin": train_cache["X"],
            "test_kin": test_cache["X"],
            "train_neural": neural_train,
            "test_neural": neural_test,
            "true_test": true_test,
            "prediction": prediction,
            "meta_test": test_cache["meta"].reset_index(drop=True),
        }

    if decoder != "wiener":
        raise ValueError(f"unknown decoder: {decoder}")
    history_bins = history_bins_for(WIENER_HISTORY_MS, BIN_MS)
    features_train, x_train, _ = du.make_history_features(
        train_cache["X"],
        neural_train,
        train_cache["meta"],
        history_bins,
        lag_bins=0,
    )
    features_test, x_test, meta_test = du.make_history_features(
        test_cache["X"],
        neural_test,
        test_cache["meta"],
        history_bins,
        lag_bins=0,
    )
    true_test, prediction, predicted_meta = wiener_fit_predict(
        train_cache["X"],
        neural_train,
        train_cache["meta"],
        test_cache["X"],
        neural_test,
        test_cache["meta"],
        history_bins,
    )
    if true_test is None:
        raise RuntimeError("Wiener history embedding produced no test bins")
    if len(predicted_meta) != len(meta_test):
        raise RuntimeError("Wiener metadata mismatch")
    return {
        "train_kin": x_train,
        "test_kin": x_test,
        "train_neural": features_train,
        "test_neural": features_test,
        "true_test": true_test,
        "prediction": prediction,
        "meta_test": predicted_meta.reset_index(drop=True),
    }


def decode_pair(train_cache, test_cache, decoder):
    neural_train, neural_test = align_pair(train_cache, test_cache)
    arrays = decoder_arrays(
        train_cache, test_cache, neural_train, neural_test, decoder
    )
    error = np.linalg.norm(
        arrays["prediction"] - arrays["true_test"], axis=1
    )
    train_dkin, test_dkin = distances_to_training(
        arrays["train_kin"], arrays["test_kin"], ridge=1e-6
    )
    _, test_dneu = distances_to_training(
        arrays["train_neural"], arrays["test_neural"], ridge=1e-3
    )
    support_threshold = np.nanpercentile(train_dkin, 95)
    in_support = test_dkin <= support_threshold
    meta_test = arrays["meta_test"]
    full_corr = m2_per_trial(
        arrays["true_test"], arrays["prediction"], meta_test
    )
    matched_corr = m2_per_trial(
        arrays["true_test"][in_support],
        arrays["prediction"][in_support],
        meta_test.loc[in_support].reset_index(drop=True),
    )
    return {
        "error": error,
        "Dkin": test_dkin,
        "Dneu": test_dneu,
        "in_support": in_support,
        "n_test_bins": len(in_support),
        "n_in_support_bins": int(in_support.sum()),
        "corr_full": full_corr,
        "corr_matched": matched_corr,
    }


def build_target_cache(target, sessions):
    return {
        session: build_cache_entry(
            session,
            BIN_MS,
            target,
            SMOOTHER_KW,
            EXCLUDE_TRIALS.get(session, ()),
        )
        for session in sessions
    }


def analyse_target(target, sessions_r1, sessions_r2):
    cache = build_target_cache(target, sessions_r1 + sessions_r2)
    detail_rows = []
    curve_rows = []
    for r1_session, r2_session in product(sessions_r1, sessions_r2):
        for decoder in DECODERS:
            forward = decode_pair(
                cache[r1_session], cache[r2_session], decoder
            )
            reverse = decode_pair(
                cache[r2_session], cache[r1_session], decoder
            )
            detail_rows.append({
                "target": target,
                "decoder": decoder,
                "wiener_history_ms": (
                    WIENER_HISTORY_MS if decoder == "wiener" else np.nan
                ),
                "r1_session": r1_session,
                "r2_session": r2_session,
                "full_R1R2": forward["corr_full"],
                "matched_R1R2": forward["corr_matched"],
                "full_R2R1": reverse["corr_full"],
                "matched_R2R1": reverse["corr_matched"],
                "frac_R2_OOD": float((~forward["in_support"]).mean()),
                "frac_R1_OOD": float((~reverse["in_support"]).mean()),
                "n_R2_test_bins": forward["n_test_bins"],
                "n_R2_in_support_bins": forward["n_in_support_bins"],
                "n_R1_test_bins": reverse["n_test_bins"],
                "n_R1_in_support_bins": reverse["n_in_support_bins"],
                "err_corr_Dkin": safe_corr(
                    forward["error"], forward["Dkin"]
                ),
                "err_corr_Dneu": safe_corr(
                    forward["error"], forward["Dneu"]
                ),
            })
            percentile = pd.Series(forward["Dkin"]).rank(pct=True).to_numpy()
            curve_rows.extend({
                "target": target,
                "decoder": decoder,
                "source": "R1→R2",
                "z_error": z_error,
                "distance_percentile": rank,
            } for z_error, rank in zip(zscore(forward["error"]), percentile))

    within_correlations = {decoder: [] for decoder in DECODERS}
    for train_session, test_session in product(sessions_r1, repeat=2):
        if train_session == test_session:
            continue
        for decoder in DECODERS:
            baseline = decode_pair(
                cache[train_session], cache[test_session], decoder
            )
            within_correlations[decoder].append(
                safe_corr(baseline["error"], baseline["Dkin"])
            )
            percentile = pd.Series(baseline["Dkin"]).rank(
                pct=True
            ).to_numpy()
            curve_rows.extend({
                "target": target,
                "decoder": decoder,
                "source": "within R1",
                "z_error": z_error,
                "distance_percentile": rank,
            } for z_error, rank in zip(
                zscore(baseline["error"]), percentile
            ))

    detail = pd.DataFrame(detail_rows)
    summary_rows = []
    for decoder in DECODERS:
        group = detail[detail["decoder"] == decoder]
        means = group.mean(numeric_only=True)
        within = float(np.nanmean(within_correlations[decoder]))
        full_gap = means.full_R2R1 - means.full_R1R2
        matched_gap = means.matched_R2R1 - means.matched_R1R2
        summary_rows.append({
            "target": target,
            "decoder": decoder,
            "wiener_history_ms": (
                WIENER_HISTORY_MS if decoder == "wiener" else np.nan
            ),
            "n_cross_epoch_pairs": len(group),
            "err_corr_Dkin_R1R2": means.err_corr_Dkin,
            "err_corr_Dkin_within_R1": within,
            "err_corr_Dkin_excess": means.err_corr_Dkin - within,
            "err_corr_Dneu_R1R2": means.err_corr_Dneu,
            "frac_R2_OOD": means.frac_R2_OOD,
            "frac_R1_OOD": means.frac_R1_OOD,
            "mean_R2_test_bins": means.n_R2_test_bins,
            "mean_R2_in_support_bins": means.n_R2_in_support_bins,
            "mean_R1_test_bins": means.n_R1_test_bins,
            "mean_R1_in_support_bins": means.n_R1_in_support_bins,
            "full_R1R2": means.full_R1R2,
            "full_R2R1": means.full_R2R1,
            "full_asymmetry": full_gap,
            "matched_R1R2": means.matched_R1R2,
            "matched_R2R1": means.matched_R2R1,
            "matched_asymmetry": matched_gap,
            "fraction_asymmetry_closed": (
                1 - matched_gap / full_gap
                if abs(full_gap) >= 0.03 else np.nan
            ),
        })
    return detail, pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def binned_curve(frame, bins):
    indices = np.clip(
        np.digitize(frame["distance_percentile"], bins) - 1,
        0,
        len(bins) - 2,
    )
    grouped = frame.assign(bin_index=indices).groupby("bin_index")["z_error"]
    index = range(len(bins) - 1)
    return (
        grouped.mean().reindex(index).to_numpy(),
        grouped.sem().reindex(index).to_numpy(),
    )


def make_figure(curves, summary, animal, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bins = np.linspace(0, 1, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    colors = {"kalman": "#2c7fb8", "wiener": "#d95f0e"}
    markers = {"kalman": "o", "wiener": "s"}
    target_order = ("relative_position", "relative_velocity")
    for row_index, target in enumerate(target_order):
        ax_curve, ax_gap = axes[row_index]
        for decoder in DECODERS:
            for source, linestyle, alpha in (
                ("R1→R2", "-", 1.0),
                ("within R1", "--", 0.7),
            ):
                subset = curves[
                    (curves["target"] == target)
                    & (curves["decoder"] == decoder)
                    & (curves["source"] == source)
                ]
                mean, sem = binned_curve(subset, bins)
                label = f"{decoder} {source}"
                ax_curve.errorbar(
                    centers,
                    mean,
                    sem,
                    color=colors[decoder],
                    marker=markers[decoder],
                    linestyle=linestyle,
                    alpha=alpha,
                    linewidth=2,
                    label=label,
                )
        ax_curve.axhline(0, color="black", linewidth=0.7)
        ax_curve.set_xlabel("kinematic Mahalanobis-to-train (percentile)")
        ax_curve.set_ylabel("decode error (z within pair)")
        ax_curve.set_title(
            f"{target.removeprefix('relative_')}: error versus distance"
        )
        ax_curve.grid(alpha=0.25)
        ax_curve.legend(fontsize=8, ncol=2)

        target_summary = summary[summary["target"] == target]
        x = np.arange(len(DECODERS))
        width = 0.34
        full = [
            target_summary.loc[
                target_summary["decoder"] == decoder, "full_asymmetry"
            ].iloc[0]
            for decoder in DECODERS
        ]
        matched = [
            target_summary.loc[
                target_summary["decoder"] == decoder, "matched_asymmetry"
            ].iloc[0]
            for decoder in DECODERS
        ]
        bars_full = ax_gap.bar(
            x - width / 2,
            full,
            width,
            color="#888888",
            edgecolor="black",
            label="full",
        )
        bars_matched = ax_gap.bar(
            x + width / 2,
            matched,
            width,
            color="#e67e22",
            edgecolor="black",
            label="in-support",
        )
        ax_gap.bar_label(bars_full, fmt="%+.3f", padding=2, fontsize=8)
        ax_gap.bar_label(bars_matched, fmt="%+.3f", padding=2, fontsize=8)
        ax_gap.axhline(0, color="black", linewidth=0.7)
        ax_gap.set_xticks(x, DECODERS)
        ax_gap.set_ylabel("R2→R1 − R1→R2 decode correlation")
        ax_gap.set_title(
            f"{target.removeprefix('relative_')}: full versus in-support gap"
        )
        ax_gap.grid(alpha=0.25, axis="y")
        ax_gap.legend(fontsize=8)

    fig.suptitle(
        f"{animal} locked-decoder Mahalanobis coverage controls "
        f"(30 ms, Butterworth order 2, lag 0 ms)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal", choices=sorted(ANIMAL_SESSIONS), default="TS"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = (
        list(sessions) for sessions in ANIMAL_SESSIONS[args.animal]
    )
    details = []
    summaries = []
    curves = []
    for target in TARGETS:
        detail, summary, curve = analyse_target(
            target, sessions_r1, sessions_r2
        )
        details.append(detail)
        summaries.append(summary)
        curves.append(curve)
    detail = pd.concat(details, ignore_index=True).assign(animal=args.animal)
    summary = pd.concat(summaries, ignore_index=True).assign(
        animal=args.animal,
        n_r1=len(sessions_r1),
        n_r2=len(sessions_r2),
    )
    curve = pd.concat(curves, ignore_index=True)
    suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    detail_path = OUT / f"coverage_controls_locked_decoders{suffix}.csv"
    summary_path = (
        OUT / f"coverage_controls_locked_decoders_summary{suffix}.csv"
    )
    figure_path = (
        FIG / f"fig_coverage_controls_locked_decoders{suffix}.png"
    )
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    make_figure(curve, summary, args.animal, figure_path)
    print(summary.to_string(index=False))
    print(f"\nsaved {detail_path}\nsaved {summary_path}\nsaved {figure_path}")


if __name__ == "__main__":
    main()
