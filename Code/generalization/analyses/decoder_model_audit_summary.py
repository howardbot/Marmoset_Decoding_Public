"""Validate, summarize and plot the matched-split decoder audit."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from private_readout_crossfit_summary import summarize_pair_metrics

REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "decoder_audit"
REFERENCE = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "kalman_component_swap"
    / "component_swap_all.csv"
)
OUT_ALL = IN_DIR / "decoder_audit_all.csv"
OUT_PAIR = IN_DIR / "decoder_audit_pair_means.csv"
OUT_DAY = IN_DIR / "decoder_audit_by_r2_session.csv"
OUT_SUMMARY = IN_DIR / "decoder_audit_summary.csv"
OUT_FIGURE = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "figures"
    / "fig_decoder_model_audit.png"
)
TARGETS = ("relative_position", "relative_velocity")
N_R2 = 3
N_R1 = 14
N_REPEATS = 5
N_FOLDS = 5
N_VARIANTS = 4
SPLIT_KEYS = ["target", "r1_session", "r2_session", "repeat", "fold"]


def load_and_validate() -> pd.DataFrame:
    paths = [
        IN_DIR / f"audit_{target}_r2_{r2_index}.csv"
        for target in TARGETS
        for r2_index in range(N_R2)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing decoder-audit shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = len(TARGETS) * N_R2 * N_R1 * N_REPEATS * N_FOLDS * N_VARIANTS
    if len(frame) != expected:
        raise AssertionError(f"expected {expected} audit rows, found {len(frame)}")
    duplicate_keys = SPLIT_KEYS + ["decoder", "variant"]
    if frame.duplicated(duplicate_keys).any():
        raise AssertionError("duplicate decoder rows within an audit split")
    return frame


def add_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["gap_native"] = frame.rev_native - frame.fwd_native
    frame["gap_common"] = frame.rev_common - frame.fwd_common
    baseline = frame[
        (frame.decoder == "kalman") & (frame.variant == "original")
    ][SPLIT_KEYS + ["gap_native", "gap_common"]].rename(
        columns={
            "gap_native": "kalman_gap_native",
            "gap_common": "kalman_gap_common",
        }
    )
    frame = frame.merge(baseline, on=SPLIT_KEYS, validate="many_to_one")
    frame["gap_closed_native"] = frame.kalman_gap_native - frame.gap_native
    frame["gap_closed_common"] = frame.kalman_gap_common - frame.gap_common
    return frame


def check_original_kalman_parity(frame: pd.DataFrame) -> tuple[float, float]:
    reference = pd.read_csv(REFERENCE)
    reference = reference[
        (reference.transition_mode == "concatenated")
        & (reference.target_mask == 0)
    ][
        ["r1_session", "r2_session", "repeat", "fold", "fwd_score", "rev_score"]
    ]
    audit = frame[
        (frame.target == "relative_position")
        & (frame.decoder == "kalman")
        & (frame.variant == "original")
    ]
    merged = audit.merge(
        reference,
        on=["r1_session", "r2_session", "repeat", "fold"],
        validate="one_to_one",
    )
    if len(merged) != N_R2 * N_R1 * N_REPEATS * N_FOLDS:
        raise AssertionError("original-Kalman parity merge is incomplete")
    forward_error = float((merged.fwd_native - merged.fwd_score).abs().max())
    reverse_error = float((merged.rev_native - merged.rev_score).abs().max())
    if max(forward_error, reverse_error) > 1e-10:
        raise AssertionError(
            f"original-Kalman parity failed: {forward_error}, {reverse_error}"
        )
    return forward_error, reverse_error


def make_summaries(frame: pd.DataFrame):
    metrics = [
        "fwd_native",
        "rev_native",
        "gap_native",
        "fwd_common",
        "rev_common",
        "gap_common",
        "gap_closed_native",
        "gap_closed_common",
    ]
    pair = frame.groupby(
        ["target", "decoder", "variant", "r1_session", "r2_session"],
        as_index=False,
    )[metrics].mean()
    grouping = ["target", "decoder", "variant"]
    day_rows, summary_rows = summarize_pair_metrics(
        pair, grouping, metrics, "decoder_model_audit"
    )
    return pair, pd.DataFrame(day_rows), pd.DataFrame(summary_rows)


def _day_points(day, target, decoder, variant):
    return day[
        (day.target == target)
        & (day.decoder == decoder)
        & (day.variant == variant)
    ].sort_values("r2_session").gap_common.to_numpy()


def _plot_conditions(ax, summary, day, target, conditions, title):
    labels = [condition[0] for condition in conditions]
    rows = []
    for _, decoder, variant in conditions:
        match = summary[
            (summary.target == target)
            & (summary.decoder == decoder)
            & (summary.variant == variant)
            & (summary.metric == "gap_common")
        ]
        if len(match) != 1:
            raise AssertionError(f"missing summary row for {decoder}/{variant}")
        rows.append(match.iloc[0])
    means = np.asarray([row.cluster_mean for row in rows])
    lower = means - np.asarray([row.hier_boot_lo for row in rows])
    upper = np.asarray([row.hier_boot_hi for row in rows]) - means
    x = np.arange(len(conditions))
    ax.errorbar(x, means, yerr=[lower, upper], fmt="o", color="black", capsize=3)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for day_index in range(N_R2):
        values = []
        for _, decoder, variant in conditions:
            points = _day_points(day, target, decoder, variant)
            values.append(points[day_index])
        ax.scatter(x + (day_index - 1) * 0.08, values, s=28, color=colors[day_index])
    ax.axhline(0, color="0.35", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("R2->R1 - R1->R2 correlation")
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.25)


def plot(summary: pd.DataFrame, day: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
    decoder_conditions = [
        ("ridge", "ridge", "original"),
        ("Wiener 50 ms", "wiener", "history_2"),
        ("ARX", "arx", "original"),
        ("Kalman", "kalman", "original"),
    ]
    for axis, target, panel in zip(
        axes,
        ("relative_position", "relative_velocity"),
        ("A  Position", "B  Velocity"),
        strict=True,
    ):
        _plot_conditions(axis, summary, day, target, decoder_conditions, panel)
    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = ("R2 0828", "R2 0829", "R2 0830")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=color, label=label)
        for color, label in zip(colors, labels, strict=True)
    ]
    handles.append(
        plt.Line2D([], [], marker="o", linestyle="-", color="black",
                   label="3-day mean + sensitivity interval")
    )
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.94), fontsize=9)

    fig.suptitle(
        "Decoder audit: one preselected parameter set per model, identical held-out splits",
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    frame = add_contrasts(load_and_validate())
    parity = check_original_kalman_parity(frame)
    pair, day, summary = make_summaries(frame)
    frame.to_csv(OUT_ALL, index=False)
    pair.to_csv(OUT_PAIR, index=False)
    day.to_csv(OUT_DAY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    plot(summary, day)

    headline = summary[summary.metric == "gap_common"][
        ["target", "decoder", "variant", "cluster_mean", "hier_boot_lo",
         "hier_boot_hi", "min_r2_day", "max_r2_day"]
    ]
    print(headline.round(4).to_string(index=False))
    print(f"\noriginal-Kalman parity max errors: {parity}")
    print(f"saved {OUT_SUMMARY}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
