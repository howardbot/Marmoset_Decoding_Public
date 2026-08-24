"""Test locked position-decoding asymmetry with session-aware summaries."""
from __future__ import annotations

import argparse
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
from scipy import stats


THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
RESULT_DIR = REPO / "Results/workflows/generalization"
FIGURE_DIR = RESULT_DIR / "figures"
N_BOOTSTRAP = 100_000
SEED = 20260811


def is_r2_session(session: str) -> bool:
    return "interferenceAndInterferenceFree" in str(session)


def paired_directions(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair R1->R2 and R2->R1 values for each exact session pair."""
    data = frame[frame["correlation"].notna()].copy()
    train_r2 = data["train_session"].map(is_r2_session)
    test_r2 = data["test_session"].map(is_r2_session)
    forward = data[~train_r2 & test_r2].rename(
        columns={
            "train_session": "r1_session",
            "test_session": "r2_session",
            "correlation": "r1_to_r2",
        }
    )[["r1_session", "r2_session", "r1_to_r2"]]
    reverse = data[train_r2 & ~test_r2].rename(
        columns={
            "train_session": "r2_session",
            "test_session": "r1_session",
            "correlation": "r2_to_r1",
        }
    )[["r1_session", "r2_session", "r2_to_r1"]]
    paired = forward.merge(
        reverse, on=["r1_session", "r2_session"], validate="one_to_one"
    )
    paired["gap"] = paired["r2_to_r1"] - paired["r1_to_r2"]
    return paired.sort_values(["r1_session", "r2_session"], ignore_index=True)


def one_sample_test(values: np.ndarray) -> dict:
    """One-tailed positive-gap test plus a two-sided interval and p-value."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {
            "n": values.size,
            "mean_gap": float(values.mean()) if values.size else np.nan,
            "sd": np.nan,
            "t": np.nan,
            "df": max(0, values.size - 1),
            "p_one_sided_gt0": np.nan,
            "p_two_sided": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "positive_fraction": float(np.mean(values > 0)) if values.size else np.nan,
        }
    one = stats.ttest_1samp(values, 0.0, alternative="greater")
    two = stats.ttest_1samp(values, 0.0, alternative="two-sided")
    ci_low, ci_high = stats.t.interval(
        0.95,
        values.size - 1,
        loc=values.mean(),
        scale=stats.sem(values),
    )
    return {
        "n": values.size,
        "mean_gap": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "t": float(one.statistic),
        "df": values.size - 1,
        "p_one_sided_gt0": float(one.pvalue),
        "p_two_sided": float(two.pvalue),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "positive_fraction": float(np.mean(values > 0)),
    }


def crossed_bootstrap(
    paired: pd.DataFrame,
    value_column: str = "gap",
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> np.ndarray:
    """Independently resample R1 and R2 days from the complete crossed grid."""
    matrix = paired.pivot(
        index="r1_session", columns="r2_session", values=value_column
    ).to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("crossed bootstrap requires a complete finite grid")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        r1_index = rng.integers(0, matrix.shape[0], matrix.shape[0])
        r2_index = rng.integers(0, matrix.shape[1], matrix.shape[1])
        draws[index] = matrix[np.ix_(r1_index, r2_index)].mean()
    return draws


def add_self_controls(paired: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Add target-day within-day ceiling and two normalized gap definitions."""
    diagonal = frame[
        frame["train_session"].eq(frame["test_session"])
        & frame["correlation"].notna()
    ].set_index("train_session")["correlation"]
    out = paired.copy()
    out["self_r1"] = out["r1_session"].map(diagonal)
    out["self_r2"] = out["r2_session"].map(diagonal)
    out["self_test_gap"] = out["self_r1"] - out["self_r2"]
    out["ceiling_adjusted_gap"] = out["gap"] - out["self_test_gap"]
    out["target_normalized_gap"] = (
        out["r2_to_r1"] / out["self_r1"]
        - out["r1_to_r2"] / out["self_r2"]
    )
    return out


def summarize(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for metric in ("gap", "ceiling_adjusted_gap", "target_normalized_gap"):
        for unit, group_column in (
            ("pair_cells", None),
            ("r1_session_means", "r1_session"),
            ("r2_session_means", "r2_session"),
        ):
            values = (
                paired[metric].to_numpy()
                if group_column is None
                else paired.groupby(group_column)[metric].mean().to_numpy()
            )
            rows.append({"metric": metric, "unit": unit, **one_sample_test(values)})

    by_r2 = (
        paired.groupby("r2_session")
        .agg(
            n_r1=("gap", "size"),
            r1_to_r2=("r1_to_r2", "mean"),
            r2_to_r1=("r2_to_r1", "mean"),
            gap=("gap", "mean"),
            self_r2=("self_r2", "first"),
        )
        .reset_index()
    )
    tests = []
    for session, group in paired.groupby("r2_session"):
        tests.append({"r2_session": session, **one_sample_test(group["gap"])})
    by_r2 = by_r2.merge(
        pd.DataFrame(tests)[
            ["r2_session", "t", "df", "p_one_sided_gt0", "p_two_sided"]
        ],
        on="r2_session",
        validate="one_to_one",
    )
    return pd.DataFrame(rows), by_r2


def make_figure(paired: pd.DataFrame, summary: pd.DataFrame, animal: str, out: Path):
    r2_sessions = sorted(paired["r2_session"].unique())
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))

    ax = axes[0]
    for index, session in enumerate(r2_sessions):
        table = paired[paired["r2_session"].eq(session)]
        ax.scatter(
            table["r1_to_r2"],
            table["r2_to_r1"],
            s=48,
            alpha=0.8,
            color=colors(index),
            label=session.split("_")[0][-4:],
        )
    limits = [
        min(paired[["r1_to_r2", "r2_to_r1"]].min()) - 0.03,
        max(paired[["r1_to_r2", "r2_to_r1"]].max()) + 0.03,
    ]
    ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("R1→R2 position correlation")
    ax.set_ylabel("R2→R1 position correlation")
    ax.set_title("Paired locked-direction scores")
    ax.legend(title="R2 day")
    ax.grid(alpha=0.2)

    ax = axes[1]
    rng = np.random.default_rng(SEED)
    for index, session in enumerate(r2_sessions):
        values = paired.loc[paired["r2_session"].eq(session), "gap"].to_numpy()
        jitter = rng.normal(0, 0.045, len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=42,
            alpha=0.78,
            color=colors(index),
        )
        ax.plot(
            [index - 0.18, index + 0.18],
            [values.mean(), values.mean()],
            color="black",
            linewidth=2,
        )
    raw_test = summary[
        summary["metric"].eq("gap") & summary["unit"].eq("pair_cells")
    ].iloc[0]
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(r2_sessions)), [s.split("_")[0][-4:] for s in r2_sessions])
    ax.set_xlabel("R2 day")
    ax.set_ylabel("R2→R1 − R1→R2")
    ax.set_title(
        f"Cell-level mean={raw_test.mean_gap:+.3f}\n"
        f"one-tailed paired t p={raw_test.p_one_sided_gt0:.3f}"
    )
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle(
        f"{animal} locked position asymmetry — 30 ms, PCA-12, CCA, lag-0 Kalman",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animal", choices=("TS", "TY"), default="TY")
    args = parser.parse_args(argv)
    animal = args.animal.upper()
    suffix = animal.lower()
    input_path = RESULT_DIR / f"locked_position_transfer_long_{suffix}.csv"
    frame = pd.read_csv(input_path)
    paired = add_self_controls(paired_directions(frame), frame)
    summary, by_r2 = summarize(paired)
    summary["crossed_boot_ci95_low"] = np.nan
    summary["crossed_boot_ci95_high"] = np.nan
    summary["crossed_boot_fraction_le0"] = np.nan
    for metric in summary["metric"].unique():
        bootstrap = crossed_bootstrap(paired, value_column=metric)
        selected = summary["metric"].eq(metric)
        summary.loc[selected, "crossed_boot_ci95_low"] = np.quantile(
            bootstrap, 0.025
        )
        summary.loc[selected, "crossed_boot_ci95_high"] = np.quantile(
            bootstrap, 0.975
        )
        summary.loc[selected, "crossed_boot_fraction_le0"] = np.mean(
            bootstrap <= 0
        )

    pair_path = RESULT_DIR / f"locked_position_asymmetry_pairs_{suffix}.csv"
    summary_path = RESULT_DIR / f"locked_position_asymmetry_significance_{suffix}.csv"
    r2_path = RESULT_DIR / f"locked_position_asymmetry_by_r2_{suffix}.csv"
    figure_path = FIGURE_DIR / f"fig_locked_position_asymmetry_significance_{suffix}.png"
    paired.to_csv(pair_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_r2.to_csv(r2_path, index=False)
    make_figure(paired, summary, animal, figure_path)

    print(summary.round(6).to_string(index=False))
    print("\nBy R2 session:")
    print(by_r2.round(6).to_string(index=False))
    print(f"\nsaved {pair_path}")
    print(f"saved {summary_path}")
    print(f"saved {r2_path}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()
