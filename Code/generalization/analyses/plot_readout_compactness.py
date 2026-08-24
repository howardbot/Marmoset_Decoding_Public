"""Plot the session-level task-predictive read-out compactness test."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[2]
IN_DIR = REPO / "Results" / "manifold_geometry" / "readout_compactness"
SESSION_CSV = IN_DIR / "readout_compactness_by_session.csv"
ALL_CSV = IN_DIR / "readout_compactness_all.csv"
FIG = REPO / "Results" / "manifold_geometry" / "figures" / "fig_readout_compactness.png"

COLORS = {"R1": "#6f7678", "R2": "#d44a3a"}
RNG = np.random.default_rng(20260715)


def scatter_epochs(ax, frame, metric, ylabel):
    for index, epoch in enumerate(("R1", "R2")):
        values = frame.loc[frame.epoch == epoch, metric].to_numpy()
        jitter = RNG.normal(0.0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            color=COLORS[epoch],
            s=38,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.hlines(
            values.mean(), index - 0.23, index + 0.23,
            color=COLORS[epoch], linewidth=3, zorder=4,
        )
    ax.set_xticks([0, 1], ["R1", "R2"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def main():
    sessions = pd.read_csv(SESSION_CSV)
    all_rows = pd.read_csv(ALL_CSV)
    sessions = sessions[sessions["mode"] == "trial_matched"].copy()
    all_rows = all_rows[all_rows["mode"] == "trial_matched"].copy()

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.1))
    scatter_epochs(
        axes[0], sessions, "effective_rank",
        "task-predictive effective rank",
    )
    axes[0].set_title("A  Predictive spectrum")

    energy = sessions.melt(
        id_vars=["epoch", "session"],
        value_vars=["rank1_energy", "rank2_energy"],
        var_name="rank", value_name="energy",
    )
    for epoch_index, epoch in enumerate(("R1", "R2")):
        for rank_index, rank in enumerate(("rank1_energy", "rank2_energy")):
            values = energy.loc[
                (energy.epoch == epoch) & (energy["rank"] == rank), "energy"
            ].to_numpy()
            x = rank_index + (-0.12 if epoch == "R1" else 0.12)
            axes[1].scatter(
                np.full(len(values), x) + RNG.normal(0, 0.018, len(values)),
                values, color=COLORS[epoch], s=25, alpha=0.65,
            )
            axes[1].hlines(
                values.mean(), x - 0.09, x + 0.09,
                color=COLORS[epoch], linewidth=3,
            )
    axes[1].set_xticks([0, 1], ["rank 1", "rank 1–2"])
    axes[1].set_ylabel("cumulative predictive energy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title("B  Spectrum concentration")
    axes[1].grid(axis="y", alpha=0.25)

    rank_columns = ["corr_rank1", "corr_rank2", "corr_rank3"]
    split_session = all_rows.groupby(["epoch", "session"])[rank_columns].mean()
    for epoch in ("R1", "R2"):
        epoch_frame = split_session.loc[epoch]
        for _, row in epoch_frame.iterrows():
            axes[2].plot(
                [1, 2, 3], row.to_numpy(), color=COLORS[epoch],
                alpha=0.16 if epoch == "R1" else 0.35, linewidth=1,
            )
        axes[2].plot(
            [1, 2, 3], epoch_frame.mean().to_numpy(),
            color=COLORS[epoch], marker="o", linewidth=3, label=epoch,
        )
    axes[2].set_xticks([1, 2, 3])
    axes[2].set_xlabel("read-out rank retained")
    axes[2].set_ylabel("held-out trial correlation")
    axes[2].set_title("C  Reduced-rank decoding")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)

    scatter_epochs(
        axes[3], sessions, "corr_best_rank",
        "mean held-out best rank",
    )
    axes[3].set_ylim(1.0, 3.0)
    axes[3].set_title("D  Cross-validated rank")

    fig.suptitle(
        "R2 does not show a more compact task-predictive read-out "
        "(32 calibration + 8 evaluation trials per split)",
        fontsize=12,
    )
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180, bbox_inches="tight")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
