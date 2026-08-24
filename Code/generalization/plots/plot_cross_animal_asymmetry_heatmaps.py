"""Plot locked position-decoder transfer matrices for TS and TY.

Rows are training days, columns are test days, and color is the mean per-trial
decode correlation. The red diagonal marks within-day five-fold CV cells.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "marmoset_matplotlib"),
)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
RESULT_DIR = REPO / "Results" / "workflows" / "generalization"
OUTPUT = (
    REPO
    / "Code"
    / "generalization"
    / "docs"
    / "figures"
    / "fig_cross_animal_position_asymmetry_heatmaps.png"
)
INPUTS = {
    "TS": RESULT_DIR / "locked_position_transfer_matrix_ts.csv",
    "TY": RESULT_DIR / "locked_position_transfer_matrix_ty.csv",
}
R1_COUNTS = {"TS": 14, "TY": 11}


def short_date(date: str) -> str:
    date = str(date)
    return date[5:] if len(date) >= 10 else date


def load_matrices() -> dict[str, pd.DataFrame]:
    matrices = {
        animal: pd.read_csv(path, index_col=0)
        for animal, path in INPUTS.items()
    }
    for matrix in matrices.values():
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
    return matrices


def block_summary(matrix: pd.DataFrame, r1_count: int) -> tuple[float, float, float]:
    values = matrix.to_numpy(dtype=float)
    forward = float(np.nanmean(values[:r1_count, r1_count:]))
    reverse = float(np.nanmean(values[r1_count:, :r1_count]))
    return forward, reverse, reverse - forward


def draw_matrix(ax, matrix, animal, vmin, vmax):
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(
        values,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    dates = [short_date(value) for value in matrix.index]
    ax.set_xticks(np.arange(len(dates)))
    ax.set_yticks(np.arange(len(dates)))
    ax.set_xticklabels(dates, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticklabels(dates, fontsize=7.5)
    ax.set_xlabel("Test day")
    ax.set_ylabel("Train day")

    for index in range(len(matrix)):
        ax.add_patch(patches.Rectangle(
            (index - 0.5, index - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#d62728",
            linewidth=1.15,
        ))
    boundary = R1_COUNTS[animal]
    if boundary < len(matrix):
        ax.axvline(boundary - 0.5, color="white", linewidth=2.0)
        ax.axhline(boundary - 0.5, color="white", linewidth=2.0)

    forward, reverse, gap = block_summary(matrix, boundary)
    handedness = "right-handed" if animal == "TS" else "left-handed"
    n_r2 = len(matrix) - boundary
    ax.set_title(
        f"{animal} ({handedness}; {boundary} R1 + {n_r2} R2 days)\n"
        f"R1→R2={forward:.3f}, R2→R1={reverse:.3f}, Δ={gap:+.3f}",
        fontsize=10.5,
    )
    return image


def main():
    matrices = load_matrices()
    finite = np.concatenate([
        matrix.to_numpy(dtype=float)[np.isfinite(matrix.to_numpy(dtype=float))]
        for matrix in matrices.values()
    ])
    vmin = min(0.0, float(np.floor(np.nanmin(finite) * 20) / 20))
    vmax = max(0.5, float(np.ceil(np.nanmax(finite) * 20) / 20))

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.2))
    image = None
    for ax, animal in zip(axes, ("TS", "TY")):
        image = draw_matrix(ax, matrices[animal], animal, vmin, vmax)

    colorbar_axis = fig.add_axes([0.925, 0.22, 0.016, 0.58])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Mean per-trial position decode correlation")
    fig.suptitle(
        "Locked position-decoder transfer matrices",
        fontsize=14,
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Rows = training days; columns = test days; red diagonal = same-day 5-fold CV",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.035,
        "Shared color scale. Locked configuration: 30 ms bins, Butterworth order 2, "
        "K=12, lag 0, Kalman.\n"
        "White lines separate the R1 and R2 blocks.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.15, top=0.87, wspace=0.26)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    for animal, matrix in matrices.items():
        forward, reverse, gap = block_summary(matrix, R1_COUNTS[animal])
        print(
            f"{animal}: R1->R2={forward:.4f}, R2->R1={reverse:.4f}, "
            f"gap={gap:+.4f}, matrix={matrix.shape}"
        )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
