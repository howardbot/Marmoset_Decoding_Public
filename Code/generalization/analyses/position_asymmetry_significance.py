"""Test whether position-decoding asymmetry remains after variability matching."""
from __future__ import annotations

import argparse
import os
import sys
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
from scipy import stats

THIS = Path(__file__).resolve()
GENERALIZATION_DIR = THIS.parents[1]
if str(GENERALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERALIZATION_DIR))

from project_config import (  # noqa: E402
    INTERFERENCE_SESSIONS,
    TS_INTERFERENCE_R1,
    TS_INTERFERENCE_R2,
    TY_INTERFERENCE_R1,
    TY_INTERFERENCE_R2,
)

REPO = THIS.parents[3]
MATCHED_CSV = (
    REPO
    / "Results/manifold_geometry/pairwise_neural_variability_band_decoding_TS_position.csv"
)
SWEEP_FILES = {
    "TS": REPO / "Results/generalization/big_sweep_crossday_long.csv",
    "TY": REPO / "Results/generalization/big_sweep_crossday_long_ty.csv",
}
MATCH_TEST_CSV = (
    REPO / "Results/manifold_geometry/position_asymmetry_significance_matched_TS.csv"
)
MATCH_PAIR_CSV = (
    REPO / "Results/manifold_geometry/position_asymmetry_significance_pairs_TS.csv"
)
SWEEP_SUMMARY_CSV = (
    REPO / "Results/generalization/big_sweep_position_asymmetry_significance_summary.csv"
)
FIGURE = (
    REPO / "Results/manifold_geometry/figures/fig_position_asymmetry_significance.png"
)

CONFIG_COLUMNS = [
    "bin_size_ms",
    "smoother",
    "lag_ms",
    "decoder",
    "target_mode",
    "history_key",
]

TS_R1 = TS_INTERFERENCE_R1
TS_R2 = TS_INTERFERENCE_R2
TY_R1 = TY_INTERFERENCE_R1
TY_R2 = TY_INTERFERENCE_R2
ANIMAL_SESSIONS = INTERFERENCE_SESSIONS

N_BOOTSTRAP = 50_000
SEED = 20260810


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaN positions."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not finite.size:
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    ranks = np.arange(1, order.size + 1, dtype=float)
    ordered_q = p_values[order] * order.size / ranks
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    adjusted[order] = np.minimum(ordered_q, 1.0)
    return adjusted


def gap_test(values: np.ndarray) -> dict:
    """One-sample t-test of a prespecified positive directional gap."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_values = values.size
    if n_values < 2:
        return {
            "n": n_values,
            "mean_gap": float(np.mean(values)) if n_values else np.nan,
            "gap_sd": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_statistic": np.nan,
            "p_one_sided_gt0": np.nan,
            "p_two_sided": np.nan,
            "cohen_dz": np.nan,
            "wilcoxon_p_one_sided_gt0": np.nan,
            "positive_fraction": float(np.mean(values > 0)) if n_values else np.nan,
        }

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    standard_error = stats.sem(values)
    ci_low, ci_high = stats.t.interval(
        0.95, n_values - 1, loc=mean, scale=standard_error
    )
    one_sided = stats.ttest_1samp(values, 0.0, alternative="greater")
    two_sided = stats.ttest_1samp(values, 0.0, alternative="two-sided")
    try:
        wilcoxon_p = float(stats.wilcoxon(values, alternative="greater").pvalue)
    except ValueError:
        wilcoxon_p = np.nan
    return {
        "n": n_values,
        "mean_gap": mean,
        "gap_sd": sd,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t_statistic": float(one_sided.statistic),
        "p_one_sided_gt0": float(one_sided.pvalue),
        "p_two_sided": float(two_sided.pvalue),
        "cohen_dz": mean / sd if sd > 0 else np.nan,
        "wilcoxon_p_one_sided_gt0": wilcoxon_p,
        "positive_fraction": float(np.mean(values > 0)),
    }


def matched_pair_gaps(
    frame: pd.DataFrame, match_direction: str, score_column: str
) -> pd.DataFrame:
    """Return one directional contrast for every R1/R2 session pair."""
    selected = frame[frame["match_direction"].eq(match_direction)]
    pivot = selected.pivot(
        index=["pair_id", "r1_session", "r2_session"],
        columns="decoder_direction",
        values=score_column,
    )
    if not {"R1->R2", "R2->R1"}.issubset(pivot.columns):
        raise ValueError("both decoder directions are required")
    paired = pivot.reset_index()
    paired["gap"] = paired["R2->R1"] - paired["R1->R2"]
    return paired


def crossed_session_bootstrap(
    paired: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
) -> np.ndarray:
    """Resample R1 and R2 sessions independently, retaining their crossed grid."""
    matrix = paired.pivot(
        index="r1_session", columns="r2_session", values="gap"
    ).to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("crossed bootstrap requires a complete finite session grid")
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_r1 = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        sampled_r2 = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draws[index] = matrix[np.ix_(sampled_r1, sampled_r2)].mean()
    return draws


def summarize_matched(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test original and matched gaps at cell, R1-day and R2-day levels."""
    rng = np.random.default_rng(SEED)
    rows = []
    pair_tables = []
    for match_direction in ("trim_r2_to_r1", "trim_r1_to_r2"):
        tables = {}
        for score_name, score_column in (
            ("original", "baseline_corr"),
            ("matched", "matched_corr"),
        ):
            paired = matched_pair_gaps(frame, match_direction, score_column)
            paired.insert(0, "score", score_name)
            paired.insert(0, "match_direction", match_direction)
            pair_tables.append(paired)
            tables[score_name] = paired

            bootstrap = crossed_session_bootstrap(paired, rng)
            bootstrap_low, bootstrap_high = np.quantile(bootstrap, [0.025, 0.975])
            for unit, group_column in (
                ("pair_cells", None),
                ("r1_session_means", "r1_session"),
                ("r2_session_means", "r2_session"),
            ):
                values = (
                    paired["gap"].to_numpy()
                    if group_column is None
                    else paired.groupby(group_column)["gap"].mean().to_numpy()
                )
                row = {
                    "analysis": "directional_gap",
                    "match_direction": match_direction,
                    "score": score_name,
                    "unit": unit,
                    **gap_test(values),
                    "crossed_boot_ci95_low": float(bootstrap_low),
                    "crossed_boot_ci95_high": float(bootstrap_high),
                    "crossed_boot_fraction_le0": float(np.mean(bootstrap <= 0)),
                }
                rows.append(row)

        original = tables["original"].set_index("pair_id")["gap"]
        matched = tables["matched"].set_index("pair_id")["gap"]
        change = matched - original
        change_test = stats.ttest_1samp(change, 0.0, alternative="two-sided")
        rows.append({
            "analysis": "matched_minus_original_gap",
            "match_direction": match_direction,
            "score": "change",
            "unit": "pair_cells",
            **gap_test(change.to_numpy()),
            "p_two_sided": float(change_test.pvalue),
            "crossed_boot_ci95_low": np.nan,
            "crossed_boot_ci95_high": np.nan,
            "crossed_boot_fraction_le0": np.nan,
        })

    return pd.DataFrame(rows), pd.concat(pair_tables, ignore_index=True)


def build_sweep_pair_gaps(
    frame: pd.DataFrame,
    r1_sessions: tuple[str, ...],
    r2_sessions: tuple[str, ...],
) -> pd.DataFrame:
    """Pair forward and reverse position scores within each sweep config."""
    data = frame[frame["target_mode"].eq("relative_position")].copy()
    forward = data["train_session"].isin(r1_sessions) & data["test_session"].isin(
        r2_sessions
    )
    reverse = data["train_session"].isin(r2_sessions) & data["test_session"].isin(
        r1_sessions
    )
    data = data[forward | reverse].copy()
    forward = forward.loc[data.index]
    data["direction"] = np.where(forward, "R1->R2", "R2->R1")
    data["r1_session"] = np.where(
        data["direction"].eq("R1->R2"), data["train_session"], data["test_session"]
    )
    data["r2_session"] = np.where(
        data["direction"].eq("R1->R2"), data["test_session"], data["train_session"]
    )
    data["history_key"] = data["history_ms"].fillna(-1).astype(int)

    # Trial-excluded 0828 rows are preferred when both variants are available.
    data["outlier_preference"] = data["outlier_mode"].eq("exclude").astype(int)
    data = (
        data.sort_values("outlier_preference", ascending=False)
        .drop_duplicates(
            CONFIG_COLUMNS + ["r1_session", "r2_session", "direction"],
            keep="first",
        )
    )
    paired = (
        data.pivot(
            index=CONFIG_COLUMNS + ["r1_session", "r2_session"],
            columns="direction",
            values="M2_mean",
        )
        .reset_index()
        .dropna(subset=["R1->R2", "R2->R1"])
    )
    paired["gap"] = paired["R2->R1"] - paired["R1->R2"]
    return paired


def test_sweep_configs(paired: pd.DataFrame, animal: str) -> pd.DataFrame:
    """Test paired directional gaps separately within every parameter config."""
    rows = []
    for config, group in paired.groupby(CONFIG_COLUMNS, sort=False):
        row = {"animal": animal, **dict(zip(CONFIG_COLUMNS, config))}
        for unit, group_column in (
            ("cell", None),
            ("r1", "r1_session"),
            ("r2", "r2_session"),
        ):
            values = (
                group["gap"].to_numpy()
                if group_column is None
                else group.groupby(group_column)["gap"].mean().to_numpy()
            )
            tested = gap_test(values)
            row[f"n_{unit}"] = tested["n"]
            row[f"mean_gap_{unit}"] = tested["mean_gap"]
            row[f"t_{unit}"] = tested["t_statistic"]
            row[f"p_one_sided_{unit}"] = tested["p_one_sided_gt0"]
            row[f"p_two_sided_{unit}"] = tested["p_two_sided"]
        rows.append(row)
    tests = pd.DataFrame(rows)
    for unit in ("cell", "r1", "r2"):
        tests[f"q_bh_one_sided_{unit}"] = fdr_bh(
            tests[f"p_one_sided_{unit}"].to_numpy()
        )
    tests["locked_config"] = (
        tests["bin_size_ms"].eq(30)
        & tests["smoother"].eq("butter_o2")
        & tests["lag_ms"].eq(0)
        & tests["decoder"].eq("kalman")
    )
    return tests


def summarize_sweep(tests: pd.DataFrame) -> pd.DataFrame:
    """Count positive and multiple-testing-corrected sweep configurations."""
    rows = []
    for (animal, decoder), group in tests.groupby(["animal", "decoder"]):
        row = {
            "animal": animal,
            "decoder": decoder,
            "n_configs": len(group),
            "mean_config_gap": group["mean_gap_cell"].mean(),
            "median_config_gap": group["mean_gap_cell"].median(),
            "positive_configs": int((group["mean_gap_cell"] > 0).sum()),
            "positive_config_fraction": float((group["mean_gap_cell"] > 0).mean()),
        }
        for unit in ("cell", "r1", "r2"):
            valid = group[f"p_one_sided_{unit}"].notna()
            row[f"n_testable_{unit}"] = int(valid.sum())
            row[f"p_lt_0_05_{unit}"] = int(
                (group.loc[valid, f"p_one_sided_{unit}"] < 0.05).sum()
            )
            row[f"q_lt_0_05_{unit}"] = int(
                (group.loc[valid, f"q_bh_one_sided_{unit}"] < 0.05).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(
    match_pairs: pd.DataFrame,
    match_tests: pd.DataFrame,
    sweep_tests: pd.DataFrame,
    figure: Path = FIGURE,
    match_label: str = "neural",
) -> None:
    """Visualize matched gaps, R2-day dependence, and sweep robustness."""
    primary = match_pairs[
        match_pairs["match_direction"].eq("trim_r2_to_r1")
    ].copy()
    original = primary[primary["score"].eq("original")]
    matched = primary[primary["score"].eq("matched")]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
    ax = axes[0]
    rng = np.random.default_rng(SEED)
    for index, (label, table, color) in enumerate(
        (("Original", original, "#7f8c8d"), (f"{match_label.title()} matched", matched, "#6a3d9a"))
    ):
        jitter = rng.normal(0, 0.045, len(table))
        ax.scatter(
            np.full(len(table), index) + jitter,
            table["gap"],
            s=22,
            alpha=0.68,
            color=color,
        )
        ax.plot([index - 0.18, index + 0.18], [table["gap"].mean()] * 2, color="black", lw=2)
    matched_test = match_tests[
        match_tests["match_direction"].eq("trim_r2_to_r1")
        & match_tests["score"].eq("matched")
        & match_tests["unit"].eq("pair_cells")
    ].iloc[0]
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks([0, 1], ["Original", f"{match_label.title()}\nmatched"])
    ax.set_ylabel("Position gap: R2→R1 − R1→R2")
    ax.set_title(
        f"TS paired session cells\nmatched mean={matched_test.mean_gap:.3f}, "
        f"p={matched_test.p_two_sided:.2g}"
    )
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    r2_original = original.groupby("r2_session")["gap"].mean()
    r2_matched = matched.groupby("r2_session")["gap"].mean()
    for index, session in enumerate(r2_original.index):
        ax.plot(
            [0, 1],
            [r2_original.loc[session], r2_matched.loc[session]],
            marker="o",
            lw=1.8,
            label=session.split("_")[0][-4:],
        )
    r2_test = match_tests[
        match_tests["match_direction"].eq("trim_r2_to_r1")
        & match_tests["score"].eq("matched")
        & match_tests["unit"].eq("r2_session_means")
    ].iloc[0]
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks([0, 1], ["Original", f"{match_label.title()}\nmatched"])
    ax.set_ylabel("Mean position gap per R2 day")
    ax.set_title(f"R2-session check (n=3)\none-sided p={r2_test.p_one_sided_gt0:.3f}")
    ax.legend(title="R2 day", fontsize=8)
    ax.grid(alpha=0.2, axis="y")

    ax = axes[2]
    labels = []
    positive = []
    significant = []
    colors = []
    color_map = {"kalman": "#2c7fb8", "wiener": "#d95f0e"}
    for (animal, decoder), group in sweep_tests.groupby(["animal", "decoder"]):
        labels.append(f"{animal}\n{decoder}")
        positive.append(100 * np.mean(group["mean_gap_cell"] > 0))
        significant.append(100 * np.mean(group["q_bh_one_sided_cell"] < 0.05))
        colors.append(color_map[decoder])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, positive, width=0.36, color=colors, alpha=0.42, label="gap > 0")
    ax.bar(x + 0.18, significant, width=0.36, color=colors, label="paired t, BH q < .05")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Parameter configurations (%)")
    ax.set_title("Position big sweep\nwithin-config directional tests")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle(
        f"Position asymmetry significance after {match_label}-variability matching",
        fontsize=13,
    )
    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    """Parse matched-decoding inputs and significance output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-csv", type=Path, default=MATCHED_CSV)
    parser.add_argument("--match-test-csv", type=Path, default=MATCH_TEST_CSV)
    parser.add_argument("--match-pair-csv", type=Path, default=MATCH_PAIR_CSV)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    parser.add_argument(
        "--match-label",
        default="neural",
        help="label used in figure text; it does not alter the input data",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run matched and sweep-level tests, then save statistics and the figure."""
    args = parse_args(argv)
    matched = pd.read_csv(args.matched_csv)
    match_tests, match_pairs = summarize_matched(matched)
    args.match_test_csv.parent.mkdir(parents=True, exist_ok=True)
    match_tests.to_csv(args.match_test_csv, index=False)
    match_pairs.to_csv(args.match_pair_csv, index=False)

    all_sweep_tests = []
    for animal, input_path in SWEEP_FILES.items():
        r1_sessions, r2_sessions = ANIMAL_SESSIONS[animal]
        paired = build_sweep_pair_gaps(
            pd.read_csv(input_path), r1_sessions, r2_sessions
        )
        tests = test_sweep_configs(paired, animal)
        suffix = animal.lower()
        output = (
            REPO
            / f"Results/generalization/big_sweep_position_asymmetry_significance_{suffix}.csv"
        )
        tests.to_csv(output, index=False)
        all_sweep_tests.append(tests)

    sweep_tests = pd.concat(all_sweep_tests, ignore_index=True)
    sweep_summary = summarize_sweep(sweep_tests)
    sweep_summary.to_csv(SWEEP_SUMMARY_CSV, index=False)
    make_figure(
        match_pairs,
        match_tests,
        sweep_tests,
        figure=args.figure,
        match_label=args.match_label,
    )

    print("\nVariance-matched position tests:")
    print(
        match_tests[
            match_tests["analysis"].eq("directional_gap")
            & match_tests["score"].eq("matched")
        ][
            [
                "match_direction",
                "unit",
                "n",
                "mean_gap",
                "t_statistic",
                "p_one_sided_gt0",
                "p_two_sided",
                "crossed_boot_ci95_low",
                "crossed_boot_ci95_high",
            ]
        ].round(6).to_string(index=False)
    )
    print("\nBig-sweep position summary:")
    print(sweep_summary.round(4).to_string(index=False))
    for tests in all_sweep_tests:
        print(f"\n{tests.animal.iloc[0]} locked configuration:")
        print(
            tests[tests["locked_config"]][
                [
                    "mean_gap_cell",
                    "n_cell",
                    "p_one_sided_cell",
                    "p_two_sided_cell",
                    "p_one_sided_r1",
                    "p_one_sided_r2",
                ]
            ].round(6).to_string(index=False)
        )
    print(f"\nsaved {args.match_test_csv}")
    print(f"saved {args.match_pair_csv}")
    print(f"saved {SWEEP_SUMMARY_CSV}")
    print(f"saved {args.figure}")


if __name__ == "__main__":
    sys.exit(main())
