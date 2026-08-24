"""Plot the report-ready TY position robustness summary.

The analysis is fixed to 30 ms bins, zero neural lag, fourth-order Butterworth
smoothing, 12 PCA dimensions, and all 12 CCA dimensions. The single axis
compares the mean R1->R2 and R2->R1 decode correlations for Kalman and the
prespecified 50 ms Wiener decoder. Error bars are descriptive SEMs across the
six R1 days paired with the available R2 day.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "marmoset_matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
GENERALIZATION = THIS.parents[1]
if str(GENERALIZATION) not in sys.path:
    sys.path.insert(0, str(GENERALIZATION))

from big_sweep_phase2_crossday import ANIMAL_SESSIONS
from plotting_common import filter_locked


REPO = THIS.parents[3]
RESULTS = REPO / "Results" / "workflows" / "decoder_benchmarks"

BIN_MS = 30
LAG_MS = 0
K_PCS = 12
CCA_DIMS = 12
TARGET = "relative_position"
WIENER_HISTORY_MS = 50

DECODERS = ("kalman", "wiener")
SMOOTHER = "butter_o4"
SMOOTHER_LABEL = "Butterworth O4"
DECODER_LABELS = {
    "kalman": "Kalman",
    "wiener": f"Wiener ({WIENER_HISTORY_MS} ms)",
}
COLORS = {
    "R1->R2": "#d62728",
    "R2->R1": "#1f77b4",
}
MARKERS = {
    "R1->R2": "s",
    "R2->R1": "o",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal",
        type=str.upper,
        choices=sorted(ANIMAL_SESSIONS),
        default="TY",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def data_paths(animal: str) -> tuple[Path, Path]:
    suffix = "" if animal == "TS" else f"_{animal.lower()}"
    data = RESULTS / "generalization" / f"big_sweep_crossday_long{suffix}.csv"
    figure = (
        REPO
        / "Code"
        / "generalization"
        / "docs"
        / "figures"
        / f"fig_{animal.lower()}_asymmetry_robustness.png"
    )
    return data, figure


def fixed_slice(data: pd.DataFrame, decoder: str, smoother: str) -> pd.DataFrame:
    overrides = {
        "target_mode": TARGET,
        "bin_size_ms": BIN_MS,
        "smoother": smoother,
        "decoder": decoder,
        "lag_ms": LAG_MS,
    }
    if decoder == "wiener":
        overrides["history_ms"] = WIENER_HISTORY_MS
    return filter_locked(data, **overrides)


def directional_by_r1_day(
    data: pd.DataFrame,
    r1_sessions: tuple[str, ...],
    r2_sessions: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean over R2 days while retaining individual R1 days."""
    forward = []
    reverse = []
    for r1 in r1_sessions:
        forward.append(
            data[
                data.train_session.eq(r1)
                & data.test_session.isin(r2_sessions)
            ].M2_mean.mean()
        )
        reverse.append(
            data[
                data.train_session.isin(r2_sessions)
                & data.test_session.eq(r1)
            ].M2_mean.mean()
        )
    return np.asarray(forward, dtype=float), np.asarray(reverse, dtype=float)


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    sem = np.std(values, ddof=1) / np.sqrt(values.size) if values.size > 1 else 0.0
    return float(np.mean(values)), float(sem)


def configuration_summary(data: pd.DataFrame, animal: str) -> pd.DataFrame:
    r1_sessions, r2_sessions = ANIMAL_SESSIONS[animal]
    rows = []
    for decoder in DECODERS:
        selected = fixed_slice(data, decoder, SMOOTHER)
        if selected.empty:
            raise RuntimeError(
                f"missing fixed scan cell for {decoder=}, smoother={SMOOTHER}"
            )
        forward, reverse = directional_by_r1_day(
            selected,
            r1_sessions,
            r2_sessions,
        )
        forward_mean, forward_sem = mean_sem(forward)
        reverse_mean, reverse_sem = mean_sem(reverse)
        rows.append(
            {
                "decoder": decoder,
                "smoother": SMOOTHER,
                "forward_mean": forward_mean,
                "forward_sem": forward_sem,
                "reverse_mean": reverse_mean,
                "reverse_sem": reverse_sem,
                "gap": float(np.nanmean(reverse - forward)),
                "positive_days": int(np.sum(reverse > forward)),
                "n_days": int(
                    np.sum(np.isfinite(reverse) & np.isfinite(forward))
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figure(data: pd.DataFrame, animal: str, output: Path):
    if "animal" in data.columns:
        data = data[data.animal.eq(animal)].copy()
    summary = configuration_summary(data, animal)

    x = np.arange(len(summary), dtype=float)
    offset = 0.10
    fig, axis = plt.subplots(figsize=(8.5, 6.4))

    for position, (_, row) in zip(x, summary.iterrows()):
        axis.plot(
            [position - offset, position + offset],
            [row.forward_mean, row.reverse_mean],
            color="#b7b7b7",
            linewidth=1.4,
            zorder=1,
        )

    for direction, side in (("R1->R2", -1), ("R2->R1", 1)):
        prefix = "forward" if direction == "R1->R2" else "reverse"
        axis.errorbar(
            x + side * offset,
            summary[f"{prefix}_mean"],
            yerr=summary[f"{prefix}_sem"],
            fmt=MARKERS[direction],
            color=COLORS[direction],
            markerfacecolor=COLORS[direction],
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=8,
            capsize=4,
            elinewidth=1.7,
            linewidth=0,
            label=direction,
            zorder=3,
        )

    for position, (_, row) in zip(x, summary.iterrows()):
        annotation_y = max(
            row.forward_mean + row.forward_sem,
            row.reverse_mean + row.reverse_sem,
        ) + 0.018
        axis.text(
            position,
            annotation_y,
            f"Δ={row.gap:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    labels = [DECODER_LABELS[row.decoder] for row in summary.itertuples()]
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=9)
    axis.set_ylabel("Mean cross-day decode correlation")
    axis.set_ylim(0.20, 0.50)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    n_r1 = len(ANIMAL_SESSIONS[animal][0])
    n_r2 = len(ANIMAL_SESSIONS[animal][1])
    handedness = "left-handed" if animal == "TY" else "right-handed"
    axis.set_title(
        f"{animal} position decoding: mean directional transfer by decoder\n"
        f"{BIN_MS} ms bins · lag {LAG_MS} · {SMOOTHER_LABEL} · "
        f"PCA K={K_PCS} · CCA d={CCA_DIMS}\n"
        f"{handedness}; n(R1)={n_r1}, n(R2)={n_r2}",
        fontsize=12,
        pad=34,
    )
    fig.text(
        0.5,
        0.018,
        "Mean ± SEM across the six paired R1 days; all pairs share TY's one "
        "available R2 session, so error bars are descriptive.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.17, top=0.70)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(summary.round(4).to_string(index=False))
    print(f"saved {output}")


def main():
    args = parse_args()
    data_path, default_output = data_paths(args.animal)
    data = pd.read_csv(data_path)
    make_figure(data, args.animal, args.output or default_output)


if __name__ == "__main__":
    main()
