"""Compare the two cross-epoch decoder directions with paired dot-line plots.

The original TS interference experiment contributes a complete 14-by-3 grid
of R1/R2 session pairs.  The currently processed forget control contributes a
complete 3-by-3 grid.  For every cell, the two decoder directions are paired before the
directional gap is calculated:

    gap = corr(R2-trained -> R1-tested) - corr(R1-trained -> R2-tested)

Random fixed-40 interference means and fixed-31 forget-control means are used.
Each experiment is internally equal-N; the two sample sizes differ because
2026-06-10 has only 31 usable trials.  The 50 subset
repeats are averaged within a biological session pair and are never treated as
independent replicates.

Outputs
-------
Results/current/comparisons/interference_vs_forget/tables/paired_directional_cells.csv
Results/current/comparisons/interference_vs_forget/tables/paired_directional_tests.csv
Results/current/comparisons/interference_vs_forget/tables/directional_gap_contrast.csv
Results/current/comparisons/interference_vs_forget/figures/paired_directional.png
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="marmoset_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
RESULTS = REPO / "Results" / "workflows" / "manifold_geometry"
FIGURES = RESULTS / "figures"
CURRENT = REPO / "Results" / "current" / "comparisons" / "interference_vs_forget"
CURRENT_TABLES = CURRENT / "tables"
CURRENT_FIGURES = CURRENT / "figures"
FORGET_RESULTS = REPO / "Results" / "current" / "forget_control" / "equal_n_3x3" / "tables"

INTERFERENCE_CELLS = RESULTS / "random_fixed40_position_cells.csv"
FORGET_CELLS = FORGET_RESULTS / "forget_control_fixed31_position_cells.csv"
FORGET_DROPOUT_CLEAN_CELLS = (
    FORGET_RESULTS / "forget_control_dropout_clean_fixed31_position_cells.csv"
)
ORIGINAL_MATCHED_PAIRS = (
    RESULTS / "position_asymmetry_significance_pairs_TS.csv"
)
POSITION_MATCHED_PAIRS = {
    "position_matched_1sd": RESULTS
    / "position_asymmetry_significance_position_matched_pairs_TS.csv",
    "position_matched_2sd": RESULTS
    / "position_asymmetry_significance_position_sd2_matched_pairs_TS.csv",
    "position_matched_3sd": RESULTS
    / "position_asymmetry_significance_position_sd3_matched_pairs_TS.csv",
}

PAIR_OUTPUT = CURRENT_TABLES / "paired_directional_cells.csv"
TEST_OUTPUT = CURRENT_TABLES / "paired_directional_tests.csv"
CONTRAST_OUTPUT = CURRENT_TABLES / "directional_gap_contrast.csv"
FIGURE_OUTPUT = CURRENT_FIGURES / "paired_directional.png"

N_BOOTSTRAP = 50_000
SEED = 20260818


def load_equal_n(path: Path, condition: str) -> pd.DataFrame:
    """Load repeat-averaged equal-N scores as one row per session pair."""
    data = pd.read_csv(path)
    required = {
        "pair_id",
        "r1_session",
        "r2_session",
        "r1_date",
        "r2_date",
        "random_forward_mean",
        "random_reverse_mean",
        "random_gap_mean",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    paired = data[
        [
            "pair_id",
            "r1_session",
            "r2_session",
            "r1_date",
            "r2_date",
            "random_forward_mean",
            "random_reverse_mean",
            "random_gap_mean",
        ]
    ].rename(
        columns={
            "random_forward_mean": "R1->R2",
            "random_reverse_mean": "R2->R1",
            "random_gap_mean": "gap",
        }
    )
    paired.insert(0, "condition", condition)
    paired["r1_date"] = paired["r1_date"].astype(str)
    paired["r2_date"] = paired["r2_date"].astype(str)

    calculated_gap = paired["R2->R1"] - paired["R1->R2"]
    if not np.allclose(calculated_gap, paired["gap"], atol=1e-10):
        raise ValueError(f"saved gaps do not match directional scores in {path}")
    return paired


def load_significance_pairs(
    path: Path,
    condition: str,
    score: str,
    match_direction: str = "trim_r2_to_r1",
) -> pd.DataFrame:
    """Select one paired-score table from a variability-matching result."""
    data = pd.read_csv(path)
    selected = data.loc[
        data["match_direction"].eq(match_direction) & data["score"].eq(score)
    ].copy()
    if len(selected) != 42:
        raise ValueError(
            f"expected 42 {condition} cells in {path}, found {len(selected)}"
        )
    selected.insert(0, "condition", condition)
    selected["r1_date"] = selected["r1_session"].str.extract(r"(20\d{6})")[0]
    selected["r2_date"] = selected["r2_session"].str.extract(r"(20\d{6})")[0]
    return selected[
        [
            "condition",
            "pair_id",
            "r1_session",
            "r2_session",
            "r1_date",
            "r2_date",
            "R1->R2",
            "R2->R1",
            "gap",
        ]
    ]


def paired_tests(paired: pd.DataFrame, analysis: str) -> dict[str, float | int | str]:
    """Return two-sided paired t and Wilcoxon signed-rank tests."""
    forward = paired["R1->R2"].to_numpy(dtype=float)
    reverse = paired["R2->R1"].to_numpy(dtype=float)
    gaps = reverse - forward
    finite = np.isfinite(forward) & np.isfinite(reverse)
    forward = forward[finite]
    reverse = reverse[finite]
    gaps = gaps[finite]
    n = gaps.size

    row: dict[str, float | int | str] = {
        "analysis": analysis,
        "n_pairs": int(n),
        "mean_r1_to_r2": float(np.mean(forward)) if n else np.nan,
        "mean_r2_to_r1": float(np.mean(reverse)) if n else np.nan,
        "mean_gap": float(np.mean(gaps)) if n else np.nan,
        "gap_sd": float(np.std(gaps, ddof=1)) if n > 1 else np.nan,
        "positive_gaps": int(np.sum(gaps > 0)),
        "paired_t_statistic": np.nan,
        "paired_t_p_two_sided": np.nan,
        "wilcoxon_statistic": np.nan,
        "wilcoxon_p_two_sided": np.nan,
        "t_ci95_low": np.nan,
        "t_ci95_high": np.nan,
    }
    if n < 2:
        return row

    t_test = stats.ttest_rel(reverse, forward, alternative="two-sided")
    standard_error = stats.sem(gaps)
    ci_low, ci_high = stats.t.interval(
        0.95, n - 1, loc=np.mean(gaps), scale=standard_error
    )
    row.update(
        {
            "paired_t_statistic": float(t_test.statistic),
            "paired_t_p_two_sided": float(t_test.pvalue),
            "t_ci95_low": float(ci_low),
            "t_ci95_high": float(ci_high),
        }
    )
    try:
        wilcoxon = stats.wilcoxon(
            reverse,
            forward,
            alternative="two-sided",
            method="auto",
        )
        row["wilcoxon_statistic"] = float(wilcoxon.statistic)
        row["wilcoxon_p_two_sided"] = float(wilcoxon.pvalue)
    except ValueError:
        pass
    return row


def crossed_condition_contrast(
    interference: pd.DataFrame,
    forget: pd.DataFrame,
) -> pd.DataFrame:
    """Bootstrap the direct interference-minus-forget difference in mean gap.

    R1 and R2 dates are independently resampled within each complete crossed
    grid. This preserves shared-date dependence instead of treating all cells
    as independent replicates.
    """
    interference_matrix = interference.pivot(
        index="r1_date", columns="r2_date", values="gap"
    ).to_numpy(dtype=float)
    forget_matrix = forget.pivot(
        index="r1_date", columns="r2_date", values="gap"
    ).to_numpy(dtype=float)
    if interference_matrix.shape != (14, 3):
        raise ValueError(
            f"expected a 14-by-3 interference grid, got {interference_matrix.shape}"
        )
    if forget_matrix.shape != (3, 3):
        raise ValueError(f"expected a 3-by-3 forget grid, got {forget_matrix.shape}")
    if not np.isfinite(interference_matrix).all() or not np.isfinite(
        forget_matrix
    ).all():
        raise ValueError("condition contrast requires complete finite grids")

    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        interference_r1 = rng.integers(0, 14, size=14)
        interference_r2 = rng.integers(0, 3, size=3)
        forget_r1 = rng.integers(0, 3, size=3)
        forget_r2 = rng.integers(0, 3, size=3)
        interference_mean = interference_matrix[
            np.ix_(interference_r1, interference_r2)
        ].mean()
        forget_mean = forget_matrix[np.ix_(forget_r1, forget_r2)].mean()
        draws[index] = interference_mean - forget_mean

    observed = float(interference_matrix.mean() - forget_matrix.mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    fraction_le_zero = float(np.mean(draws <= 0))
    fraction_ge_zero = float(np.mean(draws >= 0))
    p_two_sided = min(1.0, 2.0 * min(fraction_le_zero, fraction_ge_zero))
    return pd.DataFrame(
        [
            {
                "contrast": "interference_gap_minus_forget_gap",
                "interference_grid": "14x3",
                "forget_grid": "3x3",
                "interference_mean_gap": float(interference_matrix.mean()),
                "forget_mean_gap": float(forget_matrix.mean()),
                "observed_difference": observed,
                "bootstrap_replicates": N_BOOTSTRAP,
                "bootstrap_ci95_low": float(low),
                "bootstrap_ci95_high": float(high),
                "bootstrap_fraction_le0": fraction_le_zero,
                "bootstrap_p_two_sided": p_two_sided,
                "seed": SEED,
            }
        ]
    )


def short_date(value: str) -> str:
    """Format YYYYMMDD as M/D for compact panel titles."""
    value = str(value)
    return f"{int(value[4:6])}/{int(value[6:8])}"


def p_text(value: float) -> str:
    """Format a p-value compactly without turning small values into zero."""
    if not np.isfinite(value):
        return "NA"
    return f"{value:.2e}" if value < 0.001 else f"{value:.3f}"


def plot_paired_directions(
    interference: pd.DataFrame,
    forget: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    """Draw the 14-by-3 and complete 3-by-3 paired directional comparisons."""
    colors = {"interference": "#4C78A8", "forget": "#F58518"}
    directions = ["R1->R2", "R2->R1"]
    x = np.arange(2)

    all_scores = pd.concat(
        [
            interference[directions].stack(),
            forget[directions].stack(),
        ]
    ).to_numpy(dtype=float)
    score_range = float(np.nanmax(all_scores) - np.nanmin(all_scores))
    pad = max(0.05, 0.12 * score_range)
    y_limits = (float(np.nanmin(all_scores) - pad), float(np.nanmax(all_scores) + pad))

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.0, 7.1),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.78, hspace=0.46, wspace=0.18)

    row_specs = [
        ("interference", interference, axes[0]),
        ("forget", forget, axes[1]),
    ]
    for condition, data, row_axes in row_specs:
        r2_dates = sorted(data["r2_date"].unique())
        for column, (ax, r2_date) in enumerate(zip(row_axes, r2_dates)):
            selected = data.loc[data["r2_date"].eq(r2_date)].sort_values("r1_date")
            alpha = 0.36 if condition == "interference" else 0.80
            width = 1.0 if condition == "interference" else 1.8
            for _, pair in selected.iterrows():
                values = pair[directions].to_numpy(dtype=float)
                ax.plot(x, values, color=colors[condition], alpha=alpha, linewidth=width, zorder=1)
                ax.scatter(x, values, color=colors[condition], alpha=min(1.0, alpha + 0.2), s=22, zorder=2)

            means = selected[directions].mean().to_numpy(dtype=float)
            ax.plot(x, means, color="#222222", linewidth=2.4, zorder=4)
            ax.scatter(x, means, color="#222222", s=42, zorder=5)
            ax.axhline(0, color="#D6D6D6", linewidth=0.8, zorder=0)
            ax.set_title(f"R2 day {column + 1}: {short_date(r2_date)}", fontsize=11, weight="bold")
            ax.set_xticks(x, ["R1→R2", "R2→R1"])
            ax.set_xlim(-0.35, 1.35)
            ax.set_ylim(*y_limits)
            ax.grid(axis="y", color="#E8E8E8", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)
            if column == 0:
                ax.set_ylabel("Decoder correlation (r)", fontsize=10)

    interference_test = tests.loc[
        tests["analysis"].eq("interference_fixed40")
    ].iloc[0]
    forget_test = tests.loc[tests["analysis"].eq("forget_fixed31")].iloc[0]
    fig.suptitle(
        "Paired cross-epoch decoding directions",
        fontsize=18,
        weight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.895,
        "Each line is one R1/R2 session pair; black points show the mean within each R2 day.",
        ha="center",
        fontsize=10.5,
        color="#4B4B4B",
    )
    fig.text(
        0.5,
        0.835,
        (
            "Interference experiment — fixed-40, 14×3 cells (n=42): "
            f"paired t p={p_text(float(interference_test['paired_t_p_two_sided']))}; "
            f"Wilcoxon p={p_text(float(interference_test['wilcoxon_p_two_sided']))}"
        ),
        ha="center",
        fontsize=11,
        weight="bold",
        color=colors["interference"],
    )
    fig.text(
        0.5,
        0.425,
        (
            "Forget control — fixed-31, 3×3 cells (n=9): "
            f"paired t p={p_text(float(forget_test['paired_t_p_two_sided']))}; "
            f"Wilcoxon p={p_text(float(forget_test['wilcoxon_p_two_sided']))}"
        ),
        ha="center",
        fontsize=11,
        weight="bold",
        color=colors["forget"],
    )
    fig.text(
        0.5,
        0.045,
        "Positive line slope corresponds to gap = r(R2→R1) − r(R1→R2) > 0.",
        ha="center",
        fontsize=10,
        color="#4B4B4B",
    )
    CURRENT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Run paired tests, direct condition contrast, and figure generation."""
    CURRENT_TABLES.mkdir(parents=True, exist_ok=True)
    interference = load_equal_n(INTERFERENCE_CELLS, "interference_fixed40")
    forget = load_equal_n(FORGET_CELLS, "forget_fixed31")
    pd.concat([interference, forget], ignore_index=True).to_csv(
        PAIR_OUTPUT, index=False
    )

    test_rows = [
        paired_tests(interference, "interference_fixed40"),
        paired_tests(forget, "forget_fixed31"),
    ]
    forget_dropout_clean = load_equal_n(
        FORGET_DROPOUT_CLEAN_CELLS, "forget_dropout_clean_fixed31"
    )
    test_rows.append(
        paired_tests(forget_dropout_clean, "forget_dropout_clean_fixed31")
    )
    original_pairs = load_significance_pairs(
        ORIGINAL_MATCHED_PAIRS, "original_full_trial", "original"
    )
    neural_matched_pairs = load_significance_pairs(
        ORIGINAL_MATCHED_PAIRS, "neural_variability_matched", "matched"
    )
    test_rows.extend(
        [
            paired_tests(original_pairs, "original_full_trial"),
            paired_tests(neural_matched_pairs, "neural_variability_matched"),
        ]
    )
    for analysis, path in POSITION_MATCHED_PAIRS.items():
        pairs = load_significance_pairs(path, analysis, "matched")
        test_rows.append(paired_tests(pairs, analysis))
    tests = pd.DataFrame(test_rows)
    tests.to_csv(TEST_OUTPUT, index=False)

    contrast = crossed_condition_contrast(interference, forget)
    contrast.to_csv(CONTRAST_OUTPUT, index=False)
    plot_paired_directions(interference, forget, tests)

    print(tests.to_string(index=False))
    print("\nDirect condition contrast")
    print(contrast.to_string(index=False))
    print(f"\nSaved {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
