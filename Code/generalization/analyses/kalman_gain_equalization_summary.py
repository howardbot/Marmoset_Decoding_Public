"""Validate, summarize and plot the original-Kalman gain diagnostic."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "manifold_geometry" / "kalman_gain_equalization"
AUDIT = REPO / "Results" / "manifold_geometry" / "decoder_audit" / "decoder_audit_all.csv"
OUT_ALL = IN_DIR / "gain_equalization_all.csv"
OUT_PAIR = IN_DIR / "gain_equalization_pair_means.csv"
OUT_DAY = IN_DIR / "gain_equalization_by_r2_session.csv"
OUT_SUMMARY = IN_DIR / "gain_equalization_summary.csv"
OUT_FIGURE = REPO / "Results" / "manifold_geometry" / "figures" / "fig_kalman_gain_equalization.png"

N_R2 = 3
N_R1 = 14
N_REPEATS = 5
N_FOLDS = 5
VARIANT_ORDER = [
    "original",
    "shared_W",
    "shared_Q",
    "shared_WQ",
    "mean_gain",
    "mean_gain_shared_A",
    "mean_gain_shared_H",
    "mean_gain_shared_AH",
    "mean_gain_common_center",
    "mean_gain_common_center_shared_AH",
    "swapped_gain",
]
SPLIT_KEYS = ["r1_session", "r2_session", "repeat", "fold"]
METRICS = ["fwd_native", "rev_native", "gap_native", "fwd_common", "rev_common", "gap_common"]


def load_and_validate() -> pd.DataFrame:
    paths = [IN_DIR / f"gain_equalization_r2_{index}.csv" for index in range(N_R2)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing gain-equalization shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = N_R2 * N_R1 * N_REPEATS * N_FOLDS * len(VARIANT_ORDER)
    if len(frame) != expected:
        raise AssertionError(f"expected {expected} rows, found {len(frame)}")
    if frame.duplicated(SPLIT_KEYS + ["variant"]).any():
        raise AssertionError("duplicate gain-equalization rows")
    counts = frame.groupby(SPLIT_KEYS).size()
    if not counts.eq(len(VARIANT_ORDER)).all():
        raise AssertionError("incomplete intervention sets")
    return frame


def check_baseline_parity(frame: pd.DataFrame) -> tuple[float, ...]:
    audit = pd.read_csv(AUDIT)
    audit = audit[
        (audit.target == "relative_position")
        & (audit.decoder == "kalman")
        & (audit.variant == "original")
    ][SPLIT_KEYS + ["fwd_native", "rev_native", "fwd_common", "rev_common"]]
    baseline = frame[frame.variant == "original"]
    merged = baseline.merge(audit, on=SPLIT_KEYS, suffixes=("_gain", "_audit"))
    errors = tuple(
        float((merged[f"{metric}_gain"] - merged[f"{metric}_audit"]).abs().max())
        for metric in ("fwd_native", "rev_native", "fwd_common", "rev_common")
    )
    if max(errors) > 1e-10:
        raise AssertionError(f"baseline parity failed: {errors}")
    return errors


def summarize(frame: pd.DataFrame):
    pair = frame.groupby(["variant", "r1_session", "r2_session"], as_index=False)[METRICS].mean()
    day = pair.groupby(["variant", "r2_session"], as_index=False)[METRICS].mean()
    cluster = day.groupby("variant", as_index=False)[METRICS].mean().set_index("variant")
    original = cluster.loc["original"]
    rows = []
    for variant in VARIANT_ORDER:
        current = cluster.loc[variant]
        row = {"variant": variant}
        for metric in METRICS:
            row[metric] = current[metric]
        row["native_gap_change"] = current.gap_native - original.gap_native
        row["native_gap_closed_fraction"] = (
            (original.gap_native - current.gap_native) / original.gap_native
        )
        row["common_gap_change"] = current.gap_common - original.gap_common
        row["common_gap_closed_fraction"] = (
            (original.gap_common - current.gap_common) / original.gap_common
        )
        day_values = day[day.variant == variant].set_index("r2_session")
        row["min_r2_day_gap"] = day_values.gap_native.min()
        row["max_r2_day_gap"] = day_values.gap_native.max()
        row["all_r2_days_positive"] = bool((day_values.gap_native > 0).all())
        rows.append(row)
    return pair, day, pd.DataFrame(rows)


def plot(day: pd.DataFrame, summary: pd.DataFrame):
    labels = [
        "original", "shared W", "shared Q", "shared W+Q", "mean gain",
        "+ shared A", "+ shared H", "+ shared A+H", "+ common center",
        "+ center + shared A+H", "swapped gain",
    ]
    colors = [
        "#333333", "#56B4E9", "#009E73", "#0072B2", "#E69F00",
        "#CC79A7", "#7A5195", "#5F4690", "#88CCEE", "#117733",
        "#D55E00",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")

    ordered = summary.set_index("variant").loc[VARIANT_ORDER]
    x = np.arange(len(VARIANT_ORDER))
    axes[0].bar(x, ordered.gap_native, color=colors)
    axes[0].axhline(0, color="0.5", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=28, ha="right")
    axes[0].set_ylabel("3-day mean native position gap")
    axes[0].set_title("A  Gap after covariance/gain intervention", loc="left", weight="bold")

    day_variants = [
        "original", "mean_gain", "mean_gain_shared_AH",
        "mean_gain_common_center", "mean_gain_common_center_shared_AH",
        "swapped_gain",
    ]
    day_labels = [
        "original", "mean gain", "+ shared A+H", "+ common center",
        "+ center + shared A+H", "swapped gain",
    ]
    day_colors = [colors[0], colors[4], colors[7], colors[8], colors[9], colors[10]]
    sessions = sorted(day.r2_session.unique())
    session_labels = [session.replace("TSAL", "")[:8][-4:] for session in sessions]
    offsets = np.linspace(-0.3, 0.3, len(day_variants))
    for offset, variant, label, color in zip(
        offsets, day_variants, day_labels, day_colors
    ):
        values = day[day.variant == variant].set_index("r2_session").loc[sessions].gap_native
        axes[1].plot(np.arange(len(sessions)) + offset, values, "o", label=label, color=color)
    axes[1].axhline(0, color="0.5", lw=1)
    axes[1].set_xticks(np.arange(len(sessions)))
    axes[1].set_xticklabels(session_labels)
    axes[1].set_xlabel("R2 day")
    axes[1].set_ylabel("native position gap")
    axes[1].set_title("B  Day-level consistency", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle("Original Kalman: covariance and recursive-gain equalization", weight="bold")
    fig.tight_layout()
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    frame = load_and_validate()
    parity = check_baseline_parity(frame)
    pair, day, summary = summarize(frame)
    frame.to_csv(OUT_ALL, index=False)
    pair.to_csv(OUT_PAIR, index=False)
    day.to_csv(OUT_DAY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    plot(day, summary)
    print(summary.round(6).to_string(index=False))
    print(f"\nbaseline parity max errors: {parity}")
    print(f"saved {OUT_SUMMARY}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
