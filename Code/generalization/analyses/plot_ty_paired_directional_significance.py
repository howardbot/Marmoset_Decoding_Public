"""Plot and test TY's paired cross-epoch decoding directions.

Each pale line connects the two decoder directions for one R1/R2 session
cell.  The thick black line in each panel connects the two direction means
within one R2 date.  Tests are reported at three complementary levels:

1. all 22 paired cells (descriptive because cells share R1/R2 dates),
2. the 11 R1 pairs within each R2 date, and
3. the two independent R2-date means.

Outputs
-------
Results/generalization/ty_paired_directional_significance.csv
Results/generalization/figures/fig_ty_paired_directional_significance.png
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
RESULTS = REPO / "Results" / "generalization"
FIGURES = RESULTS / "figures"

INPUT = RESULTS / "locked_position_asymmetry_pairs_ty.csv"
TEST_OUTPUT = RESULTS / "ty_paired_directional_significance.csv"
FIGURE_OUTPUT = FIGURES / "fig_ty_paired_directional_significance.png"

FORWARD = "r1_to_r2"
REVERSE = "r2_to_r1"


def paired_test_row(data: pd.DataFrame, analysis: str) -> dict[str, object]:
    """Calculate two-sided paired t and Wilcoxon tests for one set of pairs."""
    forward = data[FORWARD].to_numpy(dtype=float)
    reverse = data[REVERSE].to_numpy(dtype=float)
    finite = np.isfinite(forward) & np.isfinite(reverse)
    forward = forward[finite]
    reverse = reverse[finite]
    gap = reverse - forward

    row: dict[str, object] = {
        "analysis": analysis,
        "n_pairs": int(gap.size),
        "mean_r1_to_r2": float(np.mean(forward)),
        "mean_r2_to_r1": float(np.mean(reverse)),
        "mean_gap": float(np.mean(gap)),
        "positive_pairs": int(np.sum(gap > 0)),
        "paired_t_statistic": np.nan,
        "paired_t_p_two_sided": np.nan,
        "wilcoxon_statistic": np.nan,
        "wilcoxon_p_two_sided": np.nan,
        "t_ci95_low": np.nan,
        "t_ci95_high": np.nan,
    }
    if gap.size < 2:
        return row

    t_result = stats.ttest_rel(reverse, forward, alternative="two-sided")
    ci_low, ci_high = stats.t.interval(
        0.95,
        gap.size - 1,
        loc=np.mean(gap),
        scale=stats.sem(gap),
    )
    wilcoxon = stats.wilcoxon(
        reverse,
        forward,
        alternative="two-sided",
        method="auto",
    )
    row.update(
        {
            "paired_t_statistic": float(t_result.statistic),
            "paired_t_p_two_sided": float(t_result.pvalue),
            "wilcoxon_statistic": float(wilcoxon.statistic),
            "wilcoxon_p_two_sided": float(wilcoxon.pvalue),
            "t_ci95_low": float(ci_low),
            "t_ci95_high": float(ci_high),
        }
    )
    return row


def p_text(value: float) -> str:
    """Format p values compactly for a presentation figure."""
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def date_label(session: str) -> str:
    """Convert a TY session identifier into a short month/day label."""
    date = pd.Series([session]).str.extract(r"(20\d{6})")[0].iloc[0]
    return f"{int(date[4:6])}/{int(date[6:8])}"


def main() -> None:
    data = pd.read_csv(INPUT)
    required = {"r1_session", "r2_session", FORWARD, REVERSE, "gap"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{INPUT} is missing columns: {sorted(missing)}")

    calculated_gap = data[REVERSE] - data[FORWARD]
    if not np.allclose(calculated_gap, data["gap"], atol=1e-10):
        raise ValueError("Saved TY gaps do not match the two directional scores")

    rows = [paired_test_row(data, "all_pair_cells")]
    r2_sessions = list(dict.fromkeys(data["r2_session"]))
    for session in r2_sessions:
        subset = data.loc[data["r2_session"].eq(session)]
        rows.append(paired_test_row(subset, f"within_r2_{date_label(session)}"))

    r2_means = (
        data.groupby("r2_session", sort=False)[[FORWARD, REVERSE]]
        .mean()
        .reset_index()
    )
    rows.append(paired_test_row(r2_means, "independent_r2_date_means"))
    tests = pd.DataFrame(rows)

    FIGURES.mkdir(parents=True, exist_ok=True)
    tests.to_csv(TEST_OUTPUT, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(
        1,
        len(r2_sessions),
        figsize=(5.4 * len(r2_sessions), 7.4),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    line_color = "#4C78A8"
    mean_color = "#202020"
    all_values = data[[FORWARD, REVERSE]].to_numpy(dtype=float)
    y_min = min(0.0, float(np.nanmin(all_values)) - 0.06)
    y_max = min(1.0, float(np.nanmax(all_values)) + 0.09)

    for axis, session in zip(axes, r2_sessions):
        subset = data.loc[data["r2_session"].eq(session)].copy()
        for _, row in subset.iterrows():
            axis.plot(
                [0, 1],
                [row[FORWARD], row[REVERSE]],
                color=line_color,
                alpha=0.34,
                linewidth=1.8,
                marker="o",
                markersize=7,
                markerfacecolor=(1, 1, 1, 0.35),
                markeredgewidth=1.5,
            )

        means = subset[[FORWARD, REVERSE]].mean().to_numpy(dtype=float)
        axis.plot(
            [0, 1],
            means,
            color=mean_color,
            linewidth=5,
            marker="o",
            markersize=12,
            zorder=10,
        )
        test = tests.loc[tests["analysis"].eq(f"within_r2_{date_label(session)}")].iloc[0]
        axis.set_title(f"R2 date: {date_label(session)}", fontsize=18, pad=15)
        axis.text(
            0.5,
            0.98,
            (
                f"mean gap = {test['mean_gap']:+.3f}\n"
                f"paired t p = {p_text(float(test['paired_t_p_two_sided']))}; "
                f"Wilcoxon p = {p_text(float(test['wilcoxon_p_two_sided']))}"
            ),
            transform=axis.transAxes,
            ha="center",
            va="top",
            color="#3A3A3A",
            fontsize=12,
        )
        axis.set_xlim(-0.35, 1.35)
        axis.set_ylim(y_min, y_max)
        axis.set_xticks([0, 1], ["R1→R2", "R2→R1"])
        axis.tick_params(axis="x", labelsize=15)
        axis.grid(axis="y", color="#DADADA", linewidth=1, alpha=0.8)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("Decoder correlation (r)", fontsize=16)

    pooled = tests.loc[tests["analysis"].eq("all_pair_cells")].iloc[0]
    independent = tests.loc[
        tests["analysis"].eq("independent_r2_date_means")
    ].iloc[0]
    figure.suptitle(
        "TY: paired cross-epoch decoding directions",
        fontsize=25,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.895,
        (
            f"Locked position decoder · {data['r1_session'].nunique()} R1 × "
            f"{data['r2_session'].nunique()} R2 cells (n={len(data)})\n"
            f"All cells: mean gap {pooled['mean_gap']:+.3f}; "
            f"paired t p={p_text(float(pooled['paired_t_p_two_sided']))}; "
            f"Wilcoxon p={p_text(float(pooled['wilcoxon_p_two_sided']))}"
        ),
        ha="center",
        va="top",
        fontsize=15,
        color="#315F8D",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.045,
        (
            f"Independent R2-date means (n={len(r2_means)}): "
            f"paired t p={p_text(float(independent['paired_t_p_two_sided']))}; "
            f"Wilcoxon p={p_text(float(independent['wilcoxon_p_two_sided']))}.\n"
            "The pooled cell result is significant, but the effect is not stable across R2 dates."
        ),
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#B23A3A",
    )
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.78, wspace=0.18)
    figure.savefig(FIGURE_OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(tests.to_string(index=False))
    print(f"Saved: {TEST_OUTPUT}")
    print(f"Saved: {FIGURE_OUTPUT}")


if __name__ == "__main__":
    main()
