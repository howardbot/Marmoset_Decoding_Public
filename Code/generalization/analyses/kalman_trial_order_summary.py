"""Validate and summarize the original-Kalman trial-order falsifier."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "kalman_trial_order"
AUDIT = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "decoder_audit"
    / "decoder_audit_all.csv"
)
OUT_ALL = IN_DIR / "trial_order_all.csv"
OUT_SPLIT = IN_DIR / "trial_order_split_sensitivity.csv"
OUT_PAIR = IN_DIR / "trial_order_pair_means.csv"
OUT_DAY = IN_DIR / "trial_order_by_r2_session.csv"
OUT_SUMMARY = IN_DIR / "trial_order_summary.csv"
OUT_FIGURE = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "figures"
    / "fig_kalman_trial_order_falsifier.png"
)
N_R2 = 3
N_R1 = 14
N_REPEATS = 5
N_FOLDS = 5
N_PERMUTATIONS = 20
SPLIT_KEYS = ["r1_session", "r2_session", "repeat", "fold"]
SCORE_METRICS = [
    "fwd_native", "rev_native", "gap_native",
    "fwd_common", "rev_common", "gap_common",
]
CHANGE_METRICS = [
    "fwd_A_relative_change", "fwd_W_relative_change",
    "rev_A_relative_change", "rev_W_relative_change",
]


def load_and_validate() -> pd.DataFrame:
    paths = [IN_DIR / f"trial_order_r2_{index}.csv" for index in range(N_R2)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing trial-order shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = N_R2 * N_R1 * N_REPEATS * N_FOLDS * (N_PERMUTATIONS + 1)
    if len(frame) != expected:
        raise AssertionError(f"expected {expected} rows, found {len(frame)}")
    if frame.duplicated(SPLIT_KEYS + ["permutation"]).any():
        raise AssertionError("duplicate trial-order rows")
    counts = frame.groupby(SPLIT_KEYS).size()
    if not counts.eq(N_PERMUTATIONS + 1).all():
        raise AssertionError("incomplete permutation sets")
    return frame


def check_baseline_parity(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    audit = pd.read_csv(AUDIT)
    audit = audit[
        (audit.target == "relative_position")
        & (audit.decoder == "kalman")
        & (audit.variant == "original")
    ][SPLIT_KEYS + ["fwd_native", "rev_native", "fwd_common", "rev_common"]]
    baseline = frame[frame.permutation == -1]
    merged = baseline.merge(audit, on=SPLIT_KEYS, suffixes=("_order", "_audit"))
    errors = tuple(
        float((merged[f"{metric}_order"] - merged[f"{metric}_audit"]).abs().max())
        for metric in ("fwd_native", "rev_native", "fwd_common", "rev_common")
    )
    if max(errors) > 1e-10:
        raise AssertionError(f"baseline parity failed: {errors}")
    return errors


def split_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    original = frame[frame.permutation == -1].set_index(SPLIT_KEYS)
    random = frame[frame.permutation >= 0]
    means = random.groupby(SPLIT_KEYS)[SCORE_METRICS + CHANGE_METRICS].mean()
    std = random.groupby(SPLIT_KEYS)[SCORE_METRICS].std().add_suffix("_perm_sd")
    result = means.join(std)
    for metric in SCORE_METRICS:
        result[f"{metric}_original"] = original[metric]
        result[f"{metric}_mean_shift"] = result[metric] - original[metric]
    return result.reset_index()


def summarize(frame: pd.DataFrame, split: pd.DataFrame):
    pair = frame.groupby(
        ["permutation", "r1_session", "r2_session"], as_index=False
    )[SCORE_METRICS + CHANGE_METRICS].mean()
    day = pair.groupby(["permutation", "r2_session"], as_index=False)[
        SCORE_METRICS + CHANGE_METRICS
    ].mean()
    cluster = day.groupby("permutation", as_index=False)[
        SCORE_METRICS + CHANGE_METRICS
    ].mean()

    baseline = cluster[cluster.permutation == -1].iloc[0]
    random = cluster[cluster.permutation >= 0]
    rows = []
    for metric in SCORE_METRICS + CHANGE_METRICS:
        values = random[metric].dropna().to_numpy()
        rows.append({
            "metric": metric,
            "original": baseline[metric],
            "permuted_mean": float(np.mean(values)),
            "permuted_sd": float(np.std(values, ddof=1)),
            "permuted_min": float(np.min(values)),
            "permuted_max": float(np.max(values)),
            "permuted_p2_5": float(np.percentile(values, 2.5)),
            "permuted_p97_5": float(np.percentile(values, 97.5)),
            "mean_shift": float(np.mean(values) - baseline[metric]),
        })
    split_rows = []
    for metric in SCORE_METRICS:
        shift = split[f"{metric}_mean_shift"]
        spread = split[f"{metric}_perm_sd"]
        split_rows.append({
            "metric": f"split_{metric}",
            "original": float(split[f"{metric}_original"].mean()),
            "permuted_mean": float(split[metric].mean()),
            "permuted_sd": float(spread.mean()),
            "permuted_min": float(spread.min()),
            "permuted_max": float(spread.max()),
            "permuted_p2_5": float(np.percentile(shift, 2.5)),
            "permuted_p97_5": float(np.percentile(shift, 97.5)),
            "mean_shift": float(shift.mean()),
        })
    return pair, day, cluster, pd.DataFrame(rows + split_rows)


def plot(day: pd.DataFrame, cluster: pd.DataFrame, summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")

    baseline_day = day[day.permutation == -1].sort_values("r2_session")
    random_day = day[day.permutation >= 0].groupby("r2_session").gap_native.agg(
        ["mean", "std"]
    ).reset_index()
    x = np.arange(N_R2)
    axes[0].errorbar(
        x - 0.08, baseline_day.gap_native, fmt="o", color="black",
        label="recorded order",
    )
    axes[0].errorbar(
        x + 0.08, random_day["mean"], yerr=random_day["std"], fmt="o",
        color="#D55E00", capsize=3, label="permuted order mean ± SD",
    )
    axes[0].axhline(0, color="0.5", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["0828", "0829", "0830"])
    axes[0].set_ylabel("native position gap")
    axes[0].set_title("A  Gap by R2 day", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    random_cluster = cluster[cluster.permutation >= 0]
    axes[1].plot(
        random_cluster.permutation, random_cluster.gap_native,
        "o-", color="#0072B2", ms=4,
    )
    baseline_gap = cluster.loc[cluster.permutation == -1, "gap_native"].iloc[0]
    axes[1].axhline(baseline_gap, color="black", ls="--", label="recorded order")
    axes[1].axhline(0, color="0.5", lw=1)
    axes[1].set_xlabel("permutation replicate")
    axes[1].set_ylabel("3-day mean native gap")
    axes[1].set_title("B  Whole-trial order sensitivity", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=8)

    labels = ["forward A", "reverse A", "forward W", "reverse W"]
    metrics = [
        "fwd_A_relative_change", "rev_A_relative_change",
        "fwd_W_relative_change", "rev_W_relative_change",
    ]
    values = [
        summary.loc[summary.metric == metric, "permuted_mean"].iloc[0]
        for metric in metrics
    ]
    axes[2].bar(np.arange(len(labels)), values, color=["#56B4E9", "#0072B2", "#E69F00", "#D55E00"])
    axes[2].set_xticks(np.arange(len(labels)))
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].set_ylabel("relative Frobenius change")
    axes[2].set_title("C  Parameter sensitivity", loc="left", weight="bold")

    fig.suptitle(
        "Original Kalman: sensitivity to arbitrary calibration-trial concatenation order",
        weight="bold",
    )
    fig.tight_layout()
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    frame = load_and_validate()
    parity = check_baseline_parity(frame)
    split = split_sensitivity(frame)
    pair, day, cluster, summary = summarize(frame, split)
    frame.to_csv(OUT_ALL, index=False)
    split.to_csv(OUT_SPLIT, index=False)
    pair.to_csv(OUT_PAIR, index=False)
    day.to_csv(OUT_DAY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    plot(day, cluster, summary)
    print(summary.round(6).to_string(index=False))
    print(f"\nbaseline parity max errors: {parity}")
    print(f"saved {OUT_SUMMARY}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
