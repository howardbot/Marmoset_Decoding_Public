"""Compare locked Kalman and Wiener cross-epoch asymmetry in TS and TY.

Both animals are summarized from the completed Phase-2 cross-day sweeps using
the same configuration: 30 ms bins, Butterworth order 2, lag 0, PCA-12/CCA,
and 50 ms Wiener history.  TS uses the documented 0828 outlier exclusion with
include fallback; TY has no special exclusion rows.

This script only re-aggregates completed decoder fits; it does not refit either
animal from raw data.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="marmoset_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="marmoset_xdg_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(_THIS.parent))

from big_sweep_phase2_crossday import (  # noqa: E402
    SESSIONS_R1,
    SESSIONS_R2,
    TY_SESSIONS_R1,
    TY_SESSIONS_R2,
)
from plotting_common import filter_locked  # noqa: E402

REPO = _THIS.parents[2]
RESULTS = REPO / "Results" / "generalization"
INPUTS = {
    "TS": RESULTS / "big_sweep_crossday_long.csv",
    "TY": RESULTS / "big_sweep_crossday_long_ty.csv",
}
SESSIONS = {
    "TS": (tuple(SESSIONS_R1), tuple(SESSIONS_R2)),
    "TY": (tuple(TY_SESSIONS_R1), tuple(TY_SESSIONS_R2)),
}
TARGETS = ("relative_position", "relative_velocity")
DECODERS = ("kalman", "wiener")
OUT_PAIR = RESULTS / "kalman_wiener_crossanimal_locked_pairs.csv"
OUT_SUMMARY = RESULTS / "kalman_wiener_crossanimal_locked_summary.csv"
OUT_DAY = RESULTS / "kalman_wiener_crossanimal_locked_r2_days.csv"
OUT_FIGURE = RESULTS / "figures" / "fig_kalman_wiener_crossanimal_locked.png"


def locked_rows(frame: pd.DataFrame, target: str, decoder: str) -> pd.DataFrame:
    overrides = {
        "bin_size_ms": 30,
        "smoother": "butter_o2",
        "target_mode": target,
        "decoder": decoder,
        "lag_ms": 0,
        "outlier_mode": "exclude",
    }
    if decoder == "wiener":
        overrides["history_ms"] = 50
    return filter_locked(frame, **overrides)


def extract_pairs(
    frame: pd.DataFrame,
    animal: str,
    target: str,
    decoder: str,
    r1_sessions: tuple[str, ...],
    r2_sessions: tuple[str, ...],
) -> pd.DataFrame:
    """Return one paired forward/reverse row for every R1/R2 day pair."""
    rows = locked_rows(frame, target, decoder).set_index(
        ["train_session", "test_session"]
    )
    output = []
    for r1_session in r1_sessions:
        for r2_session in r2_sessions:
            forward = float(rows.at[(r1_session, r2_session), "M2_mean"])
            reverse = float(rows.at[(r2_session, r1_session), "M2_mean"])
            output.append(
                {
                    "animal": animal,
                    "target": target,
                    "decoder": decoder,
                    "r1_session": r1_session,
                    "r2_session": r2_session,
                    "forward_corr": forward,
                    "reverse_corr": reverse,
                    "directional_gap": reverse - forward,
                }
            )
    return pd.DataFrame(output)


def summarize(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        pairs.groupby(["animal", "target", "decoder"], as_index=False)
        .agg(
            forward_corr=("forward_corr", "mean"),
            reverse_corr=("reverse_corr", "mean"),
            directional_gap=("directional_gap", "mean"),
            pair_gap_sd=("directional_gap", "std"),
            positive_pair_fraction=("directional_gap", lambda x: np.mean(x > 0)),
            n_pairs=("directional_gap", "size"),
        )
    )
    gaps = summary.pivot(
        index=["animal", "target"], columns="decoder", values="directional_gap"
    )
    difference = (gaps["wiener"] - gaps["kalman"]).rename(
        "wiener_minus_kalman_gap"
    )
    summary = summary.merge(
        difference.reset_index(), on=["animal", "target"], validate="many_to_one"
    )
    by_day = (
        pairs.groupby(["animal", "target", "decoder", "r2_session"], as_index=False)
        .agg(
            forward_corr=("forward_corr", "mean"),
            reverse_corr=("reverse_corr", "mean"),
            directional_gap=("directional_gap", "mean"),
            n_r1_days=("directional_gap", "size"),
        )
    )
    return summary, by_day


def plot(pairs: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    labels = ("TS\nKalman", "TS\nWiener", "TY\nKalman", "TY\nWiener")
    conditions = (("TS", "kalman"), ("TS", "wiener"), ("TY", "kalman"), ("TY", "wiener"))
    colors = {"kalman": "#0072B2", "wiener": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    rng = np.random.default_rng(20260807)
    for ax, target, title in zip(
        axes,
        TARGETS,
        ("A  Position", "B  Velocity"),
        strict=True,
    ):
        means = []
        for index, (animal, decoder) in enumerate(conditions):
            row = summary[
                (summary.animal == animal)
                & (summary.target == target)
                & (summary.decoder == decoder)
            ].iloc[0]
            means.append(float(row.directional_gap))
            values = pairs[
                (pairs.animal == animal)
                & (pairs.target == target)
                & (pairs.decoder == decoder)
            ].directional_gap.to_numpy()
            jitter = rng.uniform(-0.11, 0.11, len(values))
            ax.scatter(
                index + jitter,
                values,
                s=13,
                alpha=0.28,
                color=colors[decoder],
                linewidths=0,
            )
        ax.bar(
            np.arange(4), means,
            color=[colors[decoder] for _, decoder in conditions],
            alpha=0.72,
            width=0.62,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.axhline(0.0, color="0.25", linewidth=1.0)
        ax.set_xticks(np.arange(4), labels)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylabel("R2→R1 − R1→R2 correlation")
        ax.grid(axis="y", alpha=0.22)
        for index, value in enumerate(means):
            va = "bottom" if value >= 0 else "top"
            offset = 0.008 if value >= 0 else -0.008
            ax.text(index, value + offset, f"{value:+.3f}", ha="center", va=va, fontsize=9)
    fig.suptitle(
        "Locked cross-animal decoder comparison\n"
        "30 ms, Butterworth order 2, lag 0; Wiener history 50 ms",
        weight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Dots are paired session cells (TS: 14×3; TY: 6×1) and are descriptive, "
        "not independent biological replicates.",
        ha="center",
        fontsize=8.5,
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    pair_frames = []
    for animal, path in INPUTS.items():
        frame = pd.read_csv(path)
        r1_sessions, r2_sessions = SESSIONS[animal]
        for target in TARGETS:
            for decoder in DECODERS:
                pair_frames.append(
                    extract_pairs(
                        frame,
                        animal,
                        target,
                        decoder,
                        r1_sessions,
                        r2_sessions,
                    )
                )
    pairs = pd.concat(pair_frames, ignore_index=True)
    summary, by_day = summarize(pairs)
    pairs.to_csv(OUT_PAIR, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    by_day.to_csv(OUT_DAY, index=False)
    plot(pairs, summary, OUT_FIGURE)
    print(summary.round(4).to_string(index=False))
    print(f"saved {OUT_PAIR}")
    print(f"saved {OUT_SUMMARY}")
    print(f"saved {OUT_DAY}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
