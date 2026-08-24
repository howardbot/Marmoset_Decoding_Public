"""Plot signed R2/R1 variability differences against the fixed-40 gap.

Each point is one of the 14 x 3 TS R1/R2 session pairs.  The x-axis is a
signed log ratio, log(V_R2 / V_R1), and the y-axis is the random fixed-40
directional position gap, r(R2->R1) - r(R1->R2).  Crossed-session bootstrap
intervals resample R1 and R2 dates independently.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "marmoset_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
VARIABILITY_PATH = (
    REPO
    / "Results"
    / "manifold_geometry"
    / "trial_pair_variability_TS_daily.csv"
)
CELLS_PATH = (
    REPO
    / "Results"
    / "manifold_geometry"
    / "random_fixed40_position_cells.csv"
)
OUT_DIR = REPO / "Results" / "manifold_geometry"
FIGURE_DIR = OUT_DIR / "figures"
OUT_DATA = OUT_DIR / "variability_difference_vs_fixed40_gap.csv"
OUT_STATS = OUT_DIR / "variability_difference_vs_fixed40_gap_stats.csv"
OUT_FIGURE = FIGURE_DIR / "fig_variability_difference_vs_fixed40_gap.png"

N_BOOTSTRAP = 20_000
SEED = 20260817

METRICS = {
    "neural": {
        "session_column": "neural_mean",
        "x_column": "neural_log_ratio",
        "title": "Neural variability difference",
        "xlabel": "log(neural variability R2 / R1)",
    },
    "movement": {
        "session_column": "position_mean",
        "x_column": "movement_log_ratio",
        "title": "Movement variability difference",
        "xlabel": "log(movement variability R2 / R1)",
    },
}

R2_STYLES = {
    "20250828": ("#4C78A8", "R2 8/28"),
    "20250829": ("#F58518", "R2 8/29"),
    "20250830": ("#54A24B", "R2 8/30"),
}


def prepare_data(
    variability: pd.DataFrame, cells: pd.DataFrame
) -> pd.DataFrame:
    """Join session variability with fixed-40 decoding results by date pair."""
    variability = variability.set_index("session")
    rows = []
    for cell in cells.itertuples(index=False):
        r1 = variability.loc[cell.r1_session]
        r2 = variability.loc[cell.r2_session]
        rows.append(
            {
                "pair_id": cell.pair_id,
                "r1_session": cell.r1_session,
                "r2_session": cell.r2_session,
                "r1_date": str(cell.r1_date),
                "r2_date": str(cell.r2_date),
                "neural_log_ratio": np.log(
                    float(r2.neural_mean) / float(r1.neural_mean)
                ),
                "movement_log_ratio": np.log(
                    float(r2.position_mean) / float(r1.position_mean)
                ),
                "forward": cell.random_forward_mean,
                "reverse": cell.random_reverse_mean,
                "gap": cell.random_gap_mean,
            }
        )
    return pd.DataFrame(rows)


def slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return the ordinary least-squares slope for one x/y vector."""
    return float(np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)[0])


def crossed_bootstrap(
    data: pd.DataFrame, x_column: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resample R1 and R2 dates independently and recompute slopes and rhos.

    Resampling both crossed session factors preserves the dependence among the
    42 cells that share an R1 date or an R2 date.
    """
    r1_sessions = data["r1_session"].drop_duplicates().to_numpy()
    r2_sessions = data["r2_session"].drop_duplicates().to_numpy()
    x_matrix = (
        data.pivot(index="r1_session", columns="r2_session", values=x_column)
        .loc[r1_sessions, r2_sessions]
        .to_numpy(float)
    )
    y_matrix = (
        data.pivot(index="r1_session", columns="r2_session", values="gap")
        .loc[r1_sessions, r2_sessions]
        .to_numpy(float)
    )
    slopes = np.empty(N_BOOTSTRAP, dtype=float)
    correlations = np.empty(N_BOOTSTRAP, dtype=float)
    for repeat in range(N_BOOTSTRAP):
        r1_index = rng.integers(0, len(r1_sessions), size=len(r1_sessions))
        r2_index = rng.integers(0, len(r2_sessions), size=len(r2_sessions))
        selected_x = x_matrix[np.ix_(r1_index, r2_index)].ravel()
        selected_y = y_matrix[np.ix_(r1_index, r2_index)].ravel()
        slopes[repeat] = slope(selected_x, selected_y)
        correlations[repeat] = np.corrcoef(
            rankdata(selected_x), rankdata(selected_y)
        )[0, 1]
    return slopes, correlations


def calculate_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate observed associations and crossed-session bootstrap intervals."""
    rng = np.random.default_rng(SEED)
    rows = []
    for metric, specification in METRICS.items():
        x_column = specification["x_column"]
        observed_slope = slope(data[x_column], data["gap"])
        rho = float(spearmanr(data[x_column], data["gap"]).statistic)
        slopes, correlations = crossed_bootstrap(data, x_column, rng)
        slope_low, slope_high = np.nanquantile(slopes, [0.025, 0.975])
        rho_low, rho_high = np.nanquantile(correlations, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(data),
                "slope": observed_slope,
                "slope_crossed_ci95_low": slope_low,
                "slope_crossed_ci95_high": slope_high,
                "spearman_rho": rho,
                "rho_crossed_ci95_low": rho_low,
                "rho_crossed_ci95_high": rho_high,
                "bootstrap_fraction_slope_le0": np.mean(slopes <= 0),
            }
        )
    return pd.DataFrame(rows)


def padded_limits(
    values: np.ndarray, fraction: float = 0.08, include_zero: bool = False
) -> tuple[float, float]:
    """Return plotting limits with proportional padding and optional zero."""
    low, high = np.nanmin(values), np.nanmax(values)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    margin = max((high - low) * fraction, 0.02)
    return low - margin, high + margin


def make_figure(data: pd.DataFrame) -> None:
    """Plot neural and position-trajectory variability differences versus gap."""
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharey=True)
    y_limits = padded_limits(data["gap"].to_numpy(), fraction=0.11)
    for axis, (metric, specification) in zip(axes, METRICS.items()):
        x_column = specification["x_column"]
        for r2_date, (color, label) in R2_STYLES.items():
            selected = data.loc[data["r2_date"] == r2_date]
            axis.scatter(
                selected[x_column],
                selected["gap"],
                s=64,
                color=color,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.75,
                label=label,
                zorder=3,
            )

        coefficients = np.polyfit(data[x_column], data["gap"], 1)
        x_grid = np.linspace(data[x_column].min(), data[x_column].max(), 200)
        axis.plot(
            x_grid,
            np.polyval(coefficients, x_grid),
            color="#222222",
            linewidth=2.3,
            zorder=2,
        )
        axis.axhline(0, color="#666666", linewidth=0.9, alpha=0.75, zorder=0)
        axis.axvline(0, color="#666666", linewidth=0.9, alpha=0.75, zorder=0)
        axis.set_title(specification["title"], fontsize=12.5, weight="bold")
        axis.set_xlabel(specification["xlabel"], fontsize=11)
        axis.set_xlim(
            *padded_limits(data[x_column].to_numpy(), include_zero=True)
        )
        axis.set_ylim(*y_limits)
        axis.grid(color="#BBBBBB", linewidth=0.6, alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        "Fixed-40 directional gap\nR2→R1 − R1→R2", fontsize=11
    )
    axes[0].legend(frameon=False, loc="lower left", fontsize=9.5)
    fig.suptitle(
        "Does R2-versus-R1 variability difference explain the position gap?",
        fontsize=15,
        weight="bold",
        y=1.015,
    )
    fig.text(
        0.5,
        0.945,
        "x > 0: R2 is more variable.   y > 0: the R2-trained decoder generalizes better.",
        ha="center",
        fontsize=10.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.01, 0.995, 0.91))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Build the joined table, calculate statistics, and save the figure."""
    variability = pd.read_csv(VARIABILITY_PATH)
    cells = pd.read_csv(CELLS_PATH)
    data = prepare_data(variability, cells)
    statistics = calculate_statistics(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_DATA, index=False)
    statistics.to_csv(OUT_STATS, index=False)
    make_figure(data)
    print(statistics.to_string(index=False))
    print(f"saved {OUT_DATA}")
    print(f"saved {OUT_STATS}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
