"""Relate TS session variability to random-fixed40 cross-epoch generalization.

This is a descriptive first pass that avoids treating the 42 R1-by-R2 cells as
42 independent biological replicates in the primary figure.  Generalization is
first averaged by *training session*, leaving 14 R1 and 3 R2 observations.

Variability is measured on each complete session from the saved phase-aligned
trial-pair table.  Generalization is measured from the saved random fixed-40
position analysis.  A later confirmatory analysis should recompute variability
inside the exact same random 40-trial subsets.
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as student_t


THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
VARIABILITY_PATH = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "trial_pair_variability_TS_daily.csv"
)
GENERALIZATION_PATH = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "random_fixed40_position_cells.csv"
)
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry"
FIGURE_DIR = OUT_DIR / "figures"
OUT_SESSION = OUT_DIR / "variability_vs_generalization_session.csv"
OUT_DIRECTED = OUT_DIR / "variability_mismatch_vs_generalization_directed.csv"
OUT_STATS = OUT_DIR / "variability_vs_generalization_stats.csv"
OUT_SESSION_FIGURE = FIGURE_DIR / "fig_variability_vs_fixed40_generalization.png"
OUT_MISMATCH_FIGURE = (
    FIGURE_DIR / "fig_variability_mismatch_vs_fixed40_generalization.png"
)

METRICS = {
    "neural_mean": {
        "short": "Neural variability",
        "axis": "Neural trial-pair variability (mean MSD)",
        "transform": lambda values: np.asarray(values, dtype=float),
        "scale": "linear",
    },
    "position_mean": {
        "short": "Movement variability",
        "axis": "Position trial-pair variability (mean MSD, log scale)",
        "transform": lambda values: np.log10(np.asarray(values, dtype=float)),
        "scale": "log",
    },
}

R1_COLOR = "#4C78A8"
R2_COLOR = "#F58518"
FORWARD_COLOR = "#4C78A8"
REVERSE_COLOR = "#E45756"
N_BOOTSTRAP = 5_000
SEED = 20260817


def session_generalization(cells: pd.DataFrame) -> pd.DataFrame:
    """Return one average cross-epoch score for each training session."""
    forward = (
        cells.groupby(["r1_session", "r1_date"], as_index=False)
        .agg(
            generalization=("random_forward_mean", "mean"),
            partner_sd=("random_forward_mean", "std"),
            n_partners=("r2_session", "nunique"),
        )
        .rename(columns={"r1_session": "session", "r1_date": "date"})
    )
    forward["epoch"] = "R1"
    forward["direction"] = "R1->R2"

    reverse = (
        cells.groupby(["r2_session", "r2_date"], as_index=False)
        .agg(
            generalization=("random_reverse_mean", "mean"),
            partner_sd=("random_reverse_mean", "std"),
            n_partners=("r1_session", "nunique"),
        )
        .rename(columns={"r2_session": "session", "r2_date": "date"})
    )
    reverse["epoch"] = "R2"
    reverse["direction"] = "R2->R1"
    return pd.concat([forward, reverse], ignore_index=True)


def prepare_session_table(
    variability: pd.DataFrame, cells: pd.DataFrame
) -> pd.DataFrame:
    generalization = session_generalization(cells)
    variability = variability.copy()
    variability["date"] = variability["date"].astype(str)
    generalization["date"] = generalization["date"].astype(str)
    columns = [
        "session",
        "date",
        "epoch",
        "n_trials",
        "n_units",
        "neural_mean",
        "position_mean",
    ]
    merged = generalization.merge(
        variability[columns],
        on=["session", "date", "epoch"],
        validate="one_to_one",
    )
    merged = merged.sort_values(["epoch", "date"]).reset_index(drop=True)
    merged["epoch_day_index"] = (
        merged.groupby("epoch", sort=False).cumcount() + 1
    )
    return merged


def rank_residual(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Residualize ranks of values against an intercept and covariate rank."""
    ranked_values = rankdata(values)
    ranked_covariate = rankdata(covariate)
    design = np.column_stack([np.ones(len(values)), ranked_covariate])
    coefficients = np.linalg.lstsq(design, ranked_values, rcond=None)[0]
    return ranked_values - design @ coefficients


def partial_spearman_day(
    values: np.ndarray, outcome: np.ndarray, day_index: np.ndarray
) -> tuple[float, float]:
    x_residual = rank_residual(values, day_index)
    y_residual = rank_residual(outcome, day_index)
    rho = float(np.corrcoef(x_residual, y_residual)[0, 1])
    degrees_freedom = len(values) - 3  # two variables plus one covariate
    statistic = rho * np.sqrt(degrees_freedom / max(1e-15, 1.0 - rho**2))
    p_value = float(2 * student_t.sf(abs(statistic), degrees_freedom))
    return rho, p_value


def relation_statistics(session_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, specification in METRICS.items():
        transformed = specification["transform"](session_table[metric])
        frame = session_table.assign(metric_value=transformed)
        for group_name, group in (
            ("all_sessions", frame),
            ("R1", frame.loc[frame["epoch"] == "R1"]),
            ("R2", frame.loc[frame["epoch"] == "R2"]),
        ):
            rho, p_value = spearmanr(
                group["metric_value"], group["generalization"]
            )
            rows.append(
                {
                    "analysis": "session_level_raw",
                    "metric": metric,
                    "group": group_name,
                    "n": len(group),
                    "rho": rho,
                    "p_value": p_value,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                }
            )
        r1 = frame.loc[frame["epoch"] == "R1"]
        rho, p_value = partial_spearman_day(
            r1["metric_value"].to_numpy(),
            r1["generalization"].to_numpy(),
            r1["epoch_day_index"].to_numpy(),
        )
        rows.append(
            {
                "analysis": "session_level_R1_partial_day",
                "metric": metric,
                "group": "R1",
                "n": len(r1),
                "rho": rho,
                "p_value": p_value,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
            }
        )
    return pd.DataFrame(rows)


def prepare_directed_table(
    variability: pd.DataFrame, cells: pd.DataFrame
) -> pd.DataFrame:
    variability = variability.set_index("session")
    rows = []
    for cell in cells.itertuples(index=False):
        r1 = variability.loc[cell.r1_session]
        r2 = variability.loc[cell.r2_session]
        common = {
            "pair_id": cell.pair_id,
            "r1_session": cell.r1_session,
            "r2_session": cell.r2_session,
            "r1_date": str(cell.r1_date),
            "r2_date": str(cell.r2_date),
            "neural_log_ratio_mismatch": abs(
                np.log(float(r1.neural_mean) / float(r2.neural_mean))
            ),
            "position_log_ratio_mismatch": abs(
                np.log(float(r1.position_mean) / float(r2.position_mean))
            ),
        }
        rows.extend(
            [
                {
                    **common,
                    "direction": "R1->R2",
                    "generalization": cell.random_forward_mean,
                },
                {
                    **common,
                    "direction": "R2->R1",
                    "generalization": cell.random_reverse_mean,
                },
            ]
        )
    return pd.DataFrame(rows)


def crossed_bootstrap_rho(
    frame: pd.DataFrame, x_column: str, rng: np.random.Generator
) -> tuple[float, float]:
    """Crossed R1/R2 resampling interval for a descriptive Spearman rho."""
    r1_sessions = frame["r1_session"].drop_duplicates().to_numpy()
    r2_sessions = frame["r2_session"].drop_duplicates().to_numpy()
    x_matrix = (
        frame.pivot(index="r1_session", columns="r2_session", values=x_column)
        .loc[r1_sessions, r2_sessions]
        .to_numpy(dtype=float)
    )
    y_matrix = (
        frame.pivot(
            index="r1_session", columns="r2_session", values="generalization"
        )
        .loc[r1_sessions, r2_sessions]
        .to_numpy(dtype=float)
    )
    bootstrapped = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        sampled_r1 = rng.integers(0, len(r1_sessions), size=len(r1_sessions))
        sampled_r2 = rng.integers(0, len(r2_sessions), size=len(r2_sessions))
        selected_x = x_matrix[np.ix_(sampled_r1, sampled_r2)].ravel()
        selected_y = y_matrix[np.ix_(sampled_r1, sampled_r2)].ravel()
        bootstrapped[index] = np.corrcoef(
            rankdata(selected_x), rankdata(selected_y)
        )[0, 1]
    return tuple(np.nanquantile(bootstrapped, [0.025, 0.975]))


def mismatch_statistics(directed: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    columns = {
        "neural_mean": "neural_log_ratio_mismatch",
        "position_mean": "position_log_ratio_mismatch",
    }
    for metric, x_column in columns.items():
        for direction in ("R1->R2", "R2->R1"):
            selected = directed.loc[directed["direction"] == direction]
            rho, p_value = spearmanr(
                selected[x_column], selected["generalization"]
            )
            low, high = crossed_bootstrap_rho(selected, x_column, rng)
            rows.append(
                {
                    "analysis": "pair_mismatch_crossed_bootstrap",
                    "metric": metric,
                    "group": direction,
                    "n": len(selected),
                    "rho": rho,
                    "p_value": p_value,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def regression_line(axis: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    coefficients = np.polyfit(x, y, 1)
    grid = np.linspace(np.min(x), np.max(x), 100)
    axis.plot(grid, np.polyval(coefficients, grid), color=color, linewidth=2.0)


def plot_session_relations(
    session_table: pd.DataFrame, statistics: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.1))
    r1_colors = plt.cm.Blues(np.linspace(0.42, 0.95, 14))
    for column, (metric, specification) in enumerate(METRICS.items()):
        transformed = specification["transform"](session_table[metric])
        frame = session_table.assign(metric_value=transformed)
        r1 = frame.loc[frame["epoch"] == "R1"].sort_values("epoch_day_index")
        r2 = frame.loc[frame["epoch"] == "R2"].sort_values("epoch_day_index")

        top = axes[0, column]
        top.plot(
            r1["metric_value"],
            r1["generalization"],
            color=R1_COLOR,
            linewidth=1.0,
            alpha=0.35,
            zorder=1,
        )
        for row, color in zip(r1.itertuples(index=False), r1_colors):
            top.scatter(
                row.metric_value,
                row.generalization,
                s=67,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            top.annotate(
                str(row.epoch_day_index),
                (row.metric_value, row.generalization),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#2F4F6F",
            )
        top.scatter(
            r2["metric_value"],
            r2["generalization"],
            s=85,
            marker="^",
            color=R2_COLOR,
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        for row in r2.itertuples(index=False):
            top.annotate(
                str(row.date)[-4:-2] + "/" + str(row.date)[-2:],
                (row.metric_value, row.generalization),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8.5,
                color="#8C4A05",
            )
        regression_line(
            top,
            r1["metric_value"].to_numpy(),
            r1["generalization"].to_numpy(),
            R1_COLOR,
        )
        raw = statistics.loc[
            (statistics["analysis"] == "session_level_raw")
            & (statistics["metric"] == metric)
            & (statistics["group"] == "R1")
        ].iloc[0]
        top.set_title(
            f"{specification['short']}: raw R1 relationship\n"
            f"Spearman rho={raw.rho:+.3f}, p={raw.p_value:.3g} (n=14)",
            fontsize=11,
            weight="bold",
        )
        top.set_xlabel(specification["axis"])
        if specification["scale"] == "log":
            ticks = np.array([1.5, 2, 3, 5, 8])
            top.set_xticks(np.log10(ticks), labels=[str(value) for value in ticks])
        if column == 0:
            top.set_ylabel("Mean fixed-40 cross-epoch correlation")

        bottom = axes[1, column]
        x_residual = rank_residual(
            r1["metric_value"].to_numpy(), r1["epoch_day_index"].to_numpy()
        )
        y_residual = rank_residual(
            r1["generalization"].to_numpy(), r1["epoch_day_index"].to_numpy()
        )
        for x_value, y_value, day, color in zip(
            x_residual,
            y_residual,
            r1["epoch_day_index"],
            r1_colors,
        ):
            bottom.scatter(
                x_value,
                y_value,
                s=67,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            bottom.annotate(
                str(day),
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#2F4F6F",
            )
        regression_line(bottom, x_residual, y_residual, R1_COLOR)
        bottom.axhline(0, color="#555555", linewidth=0.8, alpha=0.6)
        bottom.axvline(0, color="#555555", linewidth=0.8, alpha=0.6)
        partial = statistics.loc[
            (statistics["analysis"] == "session_level_R1_partial_day")
            & (statistics["metric"] == metric)
        ].iloc[0]
        bottom.set_title(
            f"R1 relationship after removing day rank\n"
            f"partial rho={partial.rho:+.3f}, p={partial.p_value:.3g} (n=14)",
            fontsize=11,
            weight="bold",
        )
        bottom.set_xlabel(f"Day-adjusted {specification['short'].lower()} rank")
        if column == 0:
            bottom.set_ylabel("Day-adjusted generalization rank")

        for axis in (top, bottom):
            axis.grid(color="#BBBBBB", linewidth=0.6, alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D(
            [0], [0], marker="o", color=R1_COLOR, linewidth=1.2,
            markersize=7, label="R1 training sessions (numbers = day 1-14)"
        ),
        Line2D(
            [0], [0], marker="^", color=R2_COLOR, linewidth=0,
            markersize=8, label="R2 training sessions"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Does training-session variability predict position generalization?",
        fontsize=15,
        weight="bold",
        y=0.987,
    )
    fig.text(
        0.5,
        0.948,
        "One point per training session; decoder trial count fixed at 40. "
        "Variability is estimated from the complete session.",
        ha="center",
        fontsize=10.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.025, 0.065, 0.995, 0.92))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SESSION_FIGURE, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_mismatch_relations(
    directed: pd.DataFrame, statistics: pd.DataFrame
) -> None:
    columns = {
        "neural_mean": (
            "neural_log_ratio_mismatch",
            "Neural variability mismatch |log(train/test)|",
        ),
        "position_mean": (
            "position_log_ratio_mismatch",
            "Movement variability mismatch |log(train/test)|",
        ),
    }
    styles = {
        "R1->R2": (FORWARD_COLOR, "o", "R1→R2"),
        "R2->R1": (REVERSE_COLOR, "^", "R2→R1"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), sharey=True)
    for axis, (metric, (x_column, x_label)) in zip(axes, columns.items()):
        annotation_lines = []
        for direction, (color, marker, label) in styles.items():
            selected = directed.loc[directed["direction"] == direction]
            axis.scatter(
                selected[x_column],
                selected["generalization"],
                s=48,
                marker=marker,
                color=color,
                alpha=0.67,
                edgecolor="white",
                linewidth=0.65,
                label=label,
            )
            regression_line(
                axis,
                selected[x_column].to_numpy(),
                selected["generalization"].to_numpy(),
                color,
            )
            row = statistics.loc[
                (statistics["analysis"] == "pair_mismatch_crossed_bootstrap")
                & (statistics["metric"] == metric)
                & (statistics["group"] == direction)
            ].iloc[0]
            annotation_lines.append(
                f"{label}: rho={row.rho:+.2f}, crossed CI "
                f"[{row.ci95_low:+.2f}, {row.ci95_high:+.2f}]"
            )
        axis.set_title(
            METRICS[metric]["short"] + " mismatch",
            fontsize=11.5,
            weight="bold",
        )
        axis.set_xlabel(x_label)
        axis.text(
            0.03,
            0.97,
            "\n".join(annotation_lines),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#333333",
        )
        axis.grid(color="#BBBBBB", linewidth=0.6, alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fixed-40 directional correlation")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Does train–test variability mismatch predict poorer generalization?",
        fontsize=14.5,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_MISMATCH_FIGURE, dpi=200, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def main() -> None:
    variability = pd.read_csv(VARIABILITY_PATH)
    cells = pd.read_csv(GENERALIZATION_PATH)
    session_table = prepare_session_table(variability, cells)
    directed = prepare_directed_table(variability, cells)
    session_stats = relation_statistics(session_table)
    mismatch_stats = mismatch_statistics(directed)
    statistics = pd.concat([session_stats, mismatch_stats], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session_table.to_csv(OUT_SESSION, index=False)
    directed.to_csv(OUT_DIRECTED, index=False)
    statistics.to_csv(OUT_STATS, index=False)
    plot_session_relations(session_table, statistics)
    plot_mismatch_relations(directed, statistics)

    print(statistics.to_string(index=False))
    print(f"saved {OUT_SESSION}")
    print(f"saved {OUT_DIRECTED}")
    print(f"saved {OUT_STATS}")
    print(f"saved {OUT_SESSION_FIGURE}")
    print(f"saved {OUT_MISMATCH_FIGURE}")


if __name__ == "__main__":
    main()
