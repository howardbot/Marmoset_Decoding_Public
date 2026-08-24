"""Random equal-N control for the variance-matched 14 x 3 decoder heatmap.

For each repeat and each R1/R2 day pair, independently sample 40 trials from
each session without replacement.  The same sampled subsets are used for both
transfer directions, and subset-specific PCA, CCA, and lag-0 Kalman models are
refit exactly as in the variance-matched analysis.

This isolates the consequence of reducing the data to 40 trials from the
consequence of target-informed neural+position variability selection.

Outputs
-------
Results/workflows/manifold_geometry/random_fixed40_<target>_long.csv
Results/workflows/manifold_geometry/random_fixed40_<target>_summary.csv
Results/workflows/manifold_geometry/figures/fig_random_fixed40_<target>_control.png
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="marmoset_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from big_sweep_phase2_crossday import SESSIONS_R1, SESSIONS_R2
from decode_variability_matched_crossday import (
    date_label,
    decode_direction,
    load_raw_session,
    subset_cache,
)
from position_asymmetry_significance import crossed_session_bootstrap, gap_test

REPO_ROOT = _THIS.parents[2]
OUT_DIR = REPO_ROOT / "Results" / "workflows" / "manifold_geometry"
MATCHED_CSV = OUT_DIR / "variability_matched_crossday_fixed40.csv"
OUT_LONG = OUT_DIR / "random_fixed40_velocity_long.csv"
OUT_SUMMARY = OUT_DIR / "random_fixed40_velocity_summary.csv"
OUT_FIGURE = OUT_DIR / "figures" / "fig_random_fixed40_velocity_control.png"
TARGET_MODES = ("relative_position", "relative_velocity")
DEFAULT_TARGET_MODE = "relative_velocity"
N_TRIALS = 40
N_REPEATS = 20
SEED = 20260807
INFERENCE_SEED = 20260817
N_WORKERS = min(8, max(1, (os.cpu_count() or 2) // 2))


def sample_trial_ids(
    available_trials,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample sorted unique trial IDs without replacement."""
    available = np.asarray(sorted(set(int(x) for x in available_trials)), dtype=int)
    if n_trials > len(available):
        raise ValueError(
            f"requested {n_trials} trials from only {len(available)} available"
        )
    return np.sort(rng.choice(available, size=n_trials, replace=False))


def available_trial_ids(raw: dict) -> np.ndarray:
    """Return unique trial IDs available after decoder preprocessing."""
    return raw["meta"]["trial_number"].drop_duplicates().to_numpy(dtype=int)


def run_repeat(
    repeat: int,
    seed: int,
    raw: dict[str, dict],
    n_trials: int,
    target_mode: str,
) -> pd.DataFrame:
    """Run all 42 cells bidirectionally for one deterministic random repeat."""
    rows = []
    pair_id = 0
    for r1_index, r1_session in enumerate(SESSIONS_R1):
        for r2_index, r2_session in enumerate(SESSIONS_R2):
            pair_id += 1
            # A cell-specific RNG makes results invariant to loop interruption
            # and prevents one failed cell from changing later samples.
            cell_seed = np.random.SeedSequence(
                [seed, repeat, pair_id, r1_index, r2_index]
            )
            rng1, rng2 = [np.random.default_rng(child) for child in cell_seed.spawn(2)]
            trials1 = sample_trial_ids(
                available_trial_ids(raw[r1_session]), n_trials, rng1
            )
            trials2 = sample_trial_ids(
                available_trial_ids(raw[r2_session]), n_trials, rng2
            )
            cache1 = subset_cache(raw[r1_session], trials1)
            cache2 = subset_cache(raw[r2_session], trials2)
            forward = decode_direction(cache1, cache2)
            reverse = decode_direction(cache2, cache1)
            for direction, score in (("R1->R2", forward), ("R2->R1", reverse)):
                rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "pair_id": pair_id,
                        "target_mode": target_mode,
                        "direction": direction,
                        "r1_session": r1_session,
                        "r2_session": r2_session,
                        "r1_date": date_label(r1_session),
                        "r2_date": date_label(r2_session),
                        "n_trials_each": n_trials,
                        "random_corr": score,
                    }
                )
    return pd.DataFrame(rows)


def repeat_summary(
    long_df: pd.DataFrame,
    matched: pd.DataFrame,
    target_mode: str,
) -> pd.DataFrame:
    """One forward/reverse/asymmetry row per random repeat plus references."""
    pivot = long_df.pivot_table(
        index="repeat",
        columns="direction",
        values="random_corr",
        aggfunc="mean",
    )
    summary = pivot.rename(
        columns={"R1->R2": "random_forward", "R2->R1": "random_reverse"}
    ).reset_index()
    summary["random_asymmetry"] = (
        summary["random_reverse"] - summary["random_forward"]
    )

    target = matched.loc[matched["target_mode"] == target_mode]
    matched_means = target.groupby("direction")[
        ["baseline_corr", "matched_corr"]
    ].mean()
    baseline_forward = float(matched_means.loc["R1->R2", "baseline_corr"])
    baseline_reverse = float(matched_means.loc["R2->R1", "baseline_corr"])
    matched_forward = float(matched_means.loc["R1->R2", "matched_corr"])
    matched_reverse = float(matched_means.loc["R2->R1", "matched_corr"])
    summary["baseline_forward"] = baseline_forward
    summary["baseline_reverse"] = baseline_reverse
    summary["baseline_asymmetry"] = baseline_reverse - baseline_forward
    summary["matched_forward"] = matched_forward
    summary["matched_reverse"] = matched_reverse
    summary["matched_asymmetry"] = matched_reverse - matched_forward
    return summary


def cell_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Average random-subset scores within each biological session pair."""
    long_df = long_df.copy()
    long_df["r1_date"] = long_df["r1_date"].astype(str)
    long_df["r2_date"] = long_df["r2_date"].astype(str)
    paired = long_df.pivot(
        index=[
            "repeat",
            "pair_id",
            "r1_session",
            "r2_session",
            "r1_date",
            "r2_date",
        ],
        columns="direction",
        values="random_corr",
    ).reset_index()
    paired["gap"] = paired["R2->R1"] - paired["R1->R2"]
    rows = []
    group_columns = ["pair_id", "r1_session", "r2_session", "r1_date", "r2_date"]
    for keys, group in paired.groupby(group_columns, sort=True):
        gaps = group["gap"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_repeats": len(group),
                "random_forward_mean": group["R1->R2"].mean(),
                "random_reverse_mean": group["R2->R1"].mean(),
                "random_gap_mean": gaps.mean(),
                "random_gap_sd": gaps.std(ddof=1),
                "random_gap_q025": np.quantile(gaps, 0.025),
                "random_gap_q975": np.quantile(gaps, 0.975),
                "positive_gap_fraction": np.mean(gaps > 0),
            }
        )
    return pd.DataFrame(rows)


def session_inference(cells: pd.DataFrame) -> pd.DataFrame:
    """Test repeat-averaged cell gaps without treating repeats as replicates."""
    paired = cells.rename(columns={"random_gap_mean": "gap"})[
        ["r1_session", "r2_session", "gap"]
    ]
    bootstrap = crossed_session_bootstrap(
        paired,
        np.random.default_rng(INFERENCE_SEED),
    )
    bootstrap_low, bootstrap_high = np.quantile(bootstrap, [0.025, 0.975])
    rows = []
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
        rows.append(
            {
                "unit": unit,
                **gap_test(values),
                "crossed_boot_ci95_low": bootstrap_low,
                "crossed_boot_ci95_high": bootstrap_high,
                "crossed_boot_fraction_le0": np.mean(bootstrap <= 0),
            }
        )
    return pd.DataFrame(rows)


def _cell_matrix(rows: pd.DataFrame, value: str) -> np.ndarray:
    """Pivot cell summaries into the canonical 14-by-3 session grid."""
    r1_dates = [date_label(session) for session in SESSIONS_R1]
    r2_dates = [date_label(session) for session in SESSIONS_R2]
    rows = rows.copy()
    rows["r1_date"] = rows["r1_date"].astype(str)
    rows["r2_date"] = rows["r2_date"].astype(str)
    pivot = rows.pivot(index="r1_date", columns="r2_date", values=value)
    return pivot.reindex(index=r1_dates, columns=r2_dates).to_numpy(dtype=float)


def _heatmap(ax, values: np.ndarray, title: str, vmin: float, vmax: float):
    """Draw one annotated decoder-correlation heatmap."""
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(
        range(len(SESSIONS_R2)),
        [date_label(session)[4:] for session in SESSIONS_R2],
        fontsize=8,
    )
    ax.set_yticks(
        range(len(SESSIONS_R1)),
        [date_label(session)[4:] for session in SESSIONS_R1],
        fontsize=7,
    )
    ax.set_xlabel("R2 date")
    ax.set_ylabel("R1 date")
    threshold = vmin + 0.58 * (vmax - vmin)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                f"{values[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if values[row, col] > threshold else "black",
            )
    return image


def make_figure(
    long_df: pd.DataFrame,
    summary: pd.DataFrame,
    matched: pd.DataFrame,
    output: Path,
    target_mode: str,
):
    """Compare original, random-40 mean, and variability-matched maps."""
    target = matched.loc[matched["target_mode"] == target_mode]
    # Resumed CSVs may contain integer dates from older rows and string dates
    # from newly appended rows.  Normalize before grouping so one session cell
    # cannot split into two keys that later collapse to the same plotted label.
    long_df = long_df.copy()
    long_df["r1_date"] = long_df["r1_date"].astype(str)
    long_df["r2_date"] = long_df["r2_date"].astype(str)
    random_mean = (
        long_df.groupby(["direction", "r1_date", "r2_date"], as_index=False)
        ["random_corr"]
        .mean()
    )
    panels = []
    for direction in ("R1->R2", "R2->R1"):
        reference = target.loc[target["direction"] == direction]
        random_direction = random_mean.loc[random_mean["direction"] == direction]
        panels.extend(
            [
                (_cell_matrix(reference, "baseline_corr"), f"{direction} original"),
                (_cell_matrix(random_direction, "random_corr"), f"{direction} random 40 mean"),
                (_cell_matrix(reference, "matched_corr"), f"{direction} variance-matched 40"),
            ]
        )
    all_values = np.concatenate([values.ravel() for values, _ in panels])
    vmin = min(0.0, float(np.nanmin(all_values)))
    vmax = float(np.nanquantile(all_values, 0.99))

    fig = plt.figure(figsize=(13, 18))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.55])
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    for ax, (values, title) in zip(axes, panels):
        image = _heatmap(ax, values, title, vmin, vmax)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="corr")

    ax = fig.add_subplot(grid[2, :])
    random_asym = summary["random_asymmetry"].to_numpy()
    positions = np.ones(len(random_asym))
    jitter = np.linspace(-0.12, 0.12, len(random_asym))
    ax.scatter(
        positions + jitter,
        random_asym,
        color="#7f8c8d",
        s=40,
        alpha=0.8,
        label="random fixed-40 repeats",
    )
    ax.boxplot(
        [random_asym],
        positions=[1],
        widths=0.34,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    baseline_asym = float(summary["baseline_asymmetry"].iloc[0])
    matched_asym = float(summary["matched_asymmetry"].iloc[0])
    ax.axhline(
        baseline_asym,
        color="#4C78A8",
        linewidth=2,
        label=f"original = {baseline_asym:.3f}",
    )
    ax.axhline(
        matched_asym,
        color="#F58518",
        linewidth=2,
        label=f"variance-matched = {matched_asym:.3f}",
    )
    ax.set_xlim(0.55, 1.45)
    ax.set_xticks([1], ["R2→R1 − R1→R2"])
    ax.set_ylabel("mean directional asymmetry across 42 cells")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, loc="best")
    ax.set_title(
        f"Random fixed-40 control ({len(summary)} repeats): "
        f"mean={random_asym.mean():.3f}, 95% interval="
        f"[{np.quantile(random_asym, 0.025):.3f}, {np.quantile(random_asym, 0.975):.3f}]",
        fontsize=11,
    )

    target_label = target_mode.replace("relative_", "").title()
    fig.suptitle(
        f"{target_label} cross-day decoder: trial-count control versus variability matching\n"
        "Every random and matched cell uses 40 R1 + 40 R2 trials; PCA/CCA/Kalman refit",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    """Parse target, resampling, restart, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGET_MODES, default=DEFAULT_TARGET_MODE)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--matched", type=Path, default=MATCHED_CSV)
    parser.add_argument("--output-long", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--output-cells", type=Path, default=None)
    parser.add_argument("--output-inference", type=Path, default=None)
    parser.add_argument("--output-figure", type=Path, default=None)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="ignore an existing long CSV instead of resuming completed repeats",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="rebuild summary/figure from the completed long CSV without loading NWB",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Run or resume the random fixed-N control and export all summaries."""
    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.n_trials < 3:
        raise ValueError("--n-trials must be at least 3")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    target_label = args.target.replace("relative_", "")
    output_long = args.output_long or OUT_DIR / f"random_fixed40_{target_label}_long.csv"
    output_summary = (
        args.output_summary or OUT_DIR / f"random_fixed40_{target_label}_summary.csv"
    )
    output_cells = (
        args.output_cells or OUT_DIR / f"random_fixed40_{target_label}_cells.csv"
    )
    output_inference = args.output_inference or (
        OUT_DIR / f"random_fixed40_{target_label}_inference.csv"
    )
    output_figure = args.output_figure or (
        OUT_DIR / "figures" / f"fig_random_fixed40_{target_label}_control.png"
    )
    started = time.perf_counter()
    if args.plot_only:
        long_df = pd.read_csv(output_long)
        matched = pd.read_csv(args.matched)
        summary = repeat_summary(long_df, matched, args.target)
        cells = cell_summary(long_df)
        inference = session_inference(cells)
        summary.to_csv(output_summary, index=False)
        cells.to_csv(output_cells, index=False)
        inference.to_csv(output_inference, index=False)
        make_figure(long_df, summary, matched, output_figure, args.target)
        print(f"Saved {output_summary}")
        print(f"Saved {output_cells}")
        print(f"Saved {output_inference}")
        print(f"Saved {output_figure}")
        return
    sessions = list(SESSIONS_R1) + list(SESSIONS_R2)
    print(f"Loading 17 {args.target} sessions ...", flush=True)
    raw = {session: load_raw_session(session, args.target) for session in sessions}

    if output_long.exists() and not args.restart:
        long_df = pd.read_csv(output_long)
        completed = set(long_df["repeat"].unique())
    else:
        long_df = pd.DataFrame()
        completed = set()

    pending = [repeat for repeat in range(args.repeats) if repeat not in completed]
    for repeat in sorted(completed):
        if repeat < args.repeats:
            print(f"repeat {repeat + 1}/{args.repeats}: already complete", flush=True)

    if pending:
        worker_count = min(args.workers, len(pending))
        print(
            f"Running {len(pending)} pending repeats with {worker_count} workers; "
            "BLAS limited to one thread per worker.",
            flush=True,
        )
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        run_repeat,
                        repeat,
                        args.seed,
                        raw,
                        args.n_trials,
                        args.target,
                    ): (repeat, time.perf_counter())
                    for repeat in pending
                }
                for future in as_completed(futures):
                    repeat, repeat_started = futures[future]
                    rows = future.result()
                    long_df = pd.concat([long_df, rows], ignore_index=True)
                    long_df = long_df.sort_values(
                        ["repeat", "pair_id", "direction"]
                    ).reset_index(drop=True)
                    long_df.to_csv(output_long, index=False)
                    print(
                        f"repeat {repeat + 1}/{args.repeats}: saved "
                        f"({time.perf_counter() - repeat_started:.1f}s)",
                        flush=True,
                    )

    matched = pd.read_csv(args.matched)
    summary = repeat_summary(long_df, matched, args.target)
    cells = cell_summary(long_df)
    inference = session_inference(cells)
    summary.to_csv(output_summary, index=False)
    cells.to_csv(output_cells, index=False)
    inference.to_csv(output_inference, index=False)
    make_figure(long_df, summary, matched, output_figure, args.target)
    random_asym = summary["random_asymmetry"].to_numpy()
    matched_asym = float(summary["matched_asymmetry"].iloc[0])
    lower_tail_p = (1 + np.sum(random_asym <= matched_asym)) / (len(random_asym) + 1)
    print(f"\n=== Random fixed-40 {target_label} control ===")
    print(
        f"original asymmetry={summary['baseline_asymmetry'].iloc[0]:.4f}, "
        f"random mean={random_asym.mean():.4f}, "
        f"random 95% interval=[{np.quantile(random_asym, .025):.4f}, "
        f"{np.quantile(random_asym, .975):.4f}], "
        f"variance-matched={matched_asym:.4f}"
    )
    print(f"empirical lower-tail p={lower_tail_p:.4f}")
    print(f"Elapsed: {time.perf_counter() - started:.1f}s")
    print(f"Saved {output_long}")
    print(f"Saved {output_summary}")
    print(f"Saved {output_cells}")
    print(f"Saved {output_inference}")
    print(f"Saved {output_figure}")


if __name__ == "__main__":
    main()
