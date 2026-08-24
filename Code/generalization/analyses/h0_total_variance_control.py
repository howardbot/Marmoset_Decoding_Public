"""H0 direct total-variance control for cross-day decoding asymmetry.

Hypothesis being tested:
    R2-trained decoders may generalize better simply because R2 training neural
    activity has higher variance, which can act like noise-injection
    regularization.

Main control:
    For each ordered pair, fit the usual single-trial PCA/CCA alignment. Then
    add isotropic Gaussian noise to the training day's decode canonical dims
    only if its total variance is below the test day's total variance:

        add_var_per_dim = max(trace_cov(test) - trace_cov(train), 0) / D

    This is an increase-only total-variance match. It does not claim to match
    full covariance or temporal noise structure. Pairs that already have train
    variance >= test variance are retained and marked as not needing added
    noise.

Outputs:
    Results/workflows/manifold_geometry/h0_total_variance_control_pairs.csv
    Results/workflows/manifold_geometry/h0_total_variance_control_seed_summary.csv
    Results/workflows/manifold_geometry/h0_total_variance_control.csv
    Results/workflows/manifold_geometry/figures/fig_h0_total_variance_control.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import freeze_support
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "marmoset_matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parents[1]))
sys.path.insert(0, str(THIS.parent))

import decoder_utils as du
from manifold_align import pca_neural, trial_average_pc
from dimension_sweep import align_full
from big_sweep_phase2_crossday import (
    EXCLUDE_TRIALS,
    ANIMAL_SESSIONS,
    N_PHASE_BINS,
    SESSIONS_R1,
    SESSIONS_R2,
    SMOOTH_SIGMA_MS,
    TRIAL_RESULTS,
    UNIT_QUALITIES,
    filter_trials,
    kalman_fit_predict,
    m2_per_trial,
)

warnings.filterwarnings("ignore")

BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K = 12
D = 12
N_SEEDS = 15
SEED = 20260721
TARGETS = ("relative_position", "relative_velocity")
DEFAULT_WORKERS = min(8, max(1, (os.cpu_count() or 2) // 2))

REPO = THIS.parents[2]
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry"
PAIR_CSV = OUT_DIR / "h0_total_variance_control_pairs.csv"
SEED_CSV = OUT_DIR / "h0_total_variance_control_seed_summary.csv"
SUMMARY_CSV = OUT_DIR / "h0_total_variance_control.csv"
FIG = OUT_DIR / "figures" / "fig_h0_total_variance_control.png"

_WORKER_CACHE = None
_WORKER_THREAD_LIMITER = None


def short_session(session: str) -> str:
    match = re.search(r"2025(\d{4})", session)
    return match.group(1) if match else session


def load_session(session: str, target: str, exclude=()):
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        state, activity, meta = du.build_decoder_dataset(
            nwb,
            reach,
            target,
            bin_size=BIN_MS / 1000.0,
            unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS,
            trial_window="start_to_peak",
            **SMOOTHER_KW,
        )
    finally:
        io.close()
    state, activity, meta = filter_trials(state, activity, meta, exclude)
    smoothed = du.smooth_neural_causal(
        activity, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS
    )
    pc_activity = pca_neural(smoothed, k=K)[0]
    return (
        state,
        pc_activity,
        meta,
        trial_average_pc(pc_activity, meta, n_phase_bins=N_PHASE_BINS),
    )


def decode(train_state, train_activity, test_state, test_activity, test_meta) -> float:
    state_eval, prediction = kalman_fit_predict(
        train_state,
        train_activity[:, :D],
        test_state,
        test_activity[:, :D],
        test_meta,
    )
    return m2_per_trial(state_eval, prediction, test_meta)


def total_variance(activity: np.ndarray, dims: int = D) -> float:
    """Trace of the sample covariance in the decode canonical dimensions."""
    return float(np.var(np.asarray(activity)[:, :dims], axis=0, ddof=1).sum())


def _sample_column_noise_orthogonal_to_activity(
    activity_column: np.ndarray,
    add_var: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample centered noise with exact variance and zero covariance with one dim.

    The orthogonalization makes the finite-sample variance check exact for each
    dimension: var(y + noise) = var(y) + add_var. Without this step the match is
    correct only in expectation, which makes the verification panel needlessly
    noisy.
    """
    if add_var <= 0:
        return np.zeros(len(activity_column), dtype=float)

    centered_activity = np.asarray(activity_column, dtype=float)
    centered_activity = centered_activity - centered_activity.mean()
    for _ in range(20):
        noise = rng.standard_normal(len(centered_activity))
        noise = noise - noise.mean()
        denom = float(np.dot(centered_activity, centered_activity))
        if denom > 1e-12:
            noise = noise - centered_activity * (
                float(np.dot(noise, centered_activity)) / denom
            )
            noise = noise - noise.mean()
        noise_std = float(np.std(noise, ddof=1))
        if noise_std > 1e-12:
            return noise * (np.sqrt(add_var) / noise_std)
    raise RuntimeError("could not sample a non-degenerate noise column")


def match_total_variance(
    train_activity: np.ndarray,
    test_activity: np.ndarray,
    rng: np.random.Generator,
    dims: int = D,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Increase train total variance to the test total variance when needed."""
    train_activity = np.asarray(train_activity, dtype=float)
    test_activity = np.asarray(test_activity, dtype=float)
    train_before = total_variance(train_activity, dims)
    test_total = total_variance(test_activity, dims)
    gap = test_total - train_before
    add_var_per_dim = max(gap, 0.0) / dims
    matched = train_activity.copy()
    if add_var_per_dim > 0:
        for dim in range(dims):
            matched[:, dim] = matched[:, dim] + _sample_column_noise_orthogonal_to_activity(
                matched[:, dim], add_var_per_dim, rng
            )
    train_after = total_variance(matched, dims)
    return matched, {
        "train_total_var_before": train_before,
        "train_total_var_after": train_after,
        "test_total_var": test_total,
        "target_minus_train_var": gap,
        "added_var_per_dim": add_var_per_dim,
        "needs_noise": bool(add_var_per_dim > 0),
        "variance_error_after": train_after - test_total,
    }


def evaluate_ordered_pair(
    cache: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]],
    train_session: str,
    test_session: str,
    direction: str,
    target: str,
    seed: int,
    pair_index: int,
    direction_index: int,
) -> dict[str, float | int | str | bool]:
    fit_seed = SEED + seed * 100_000 + pair_index * 1000 + direction_index * 100
    train = cache[train_session]
    test = cache[test_session]
    train_aligned, test_aligned = align_full(
        "single_trial",
        K,
        train,
        test,
        np.random.default_rng(fit_seed),
    )
    if train_aligned is None:
        raise RuntimeError(f"alignment failed for {train_session} -> {test_session}")

    baseline = decode(train[0], train_aligned, test[0], test_aligned, test[2])
    matched_train, variance_info = match_total_variance(
        train_aligned,
        test_aligned,
        np.random.default_rng(fit_seed + 1),
    )
    matched = decode(train[0], matched_train, test[0], test_aligned, test[2])
    return {
        "target": target,
        "seed": seed,
        "pair_index": pair_index,
        "direction_index": direction_index,
        "direction": direction,
        "train_session": train_session,
        "test_session": test_session,
        "train_day": short_session(train_session),
        "test_day": short_session(test_session),
        "baseline_corr": baseline,
        "matched_corr": matched,
        "delta_corr": matched - baseline,
        **variance_info,
    }


def _init_evaluation_worker(cache):
    """Install one read-only cache per worker and prevent BLAS oversubscription."""
    global _WORKER_CACHE, _WORKER_THREAD_LIMITER
    _WORKER_CACHE = cache
    _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)


def _evaluate_task(task):
    if _WORKER_CACHE is None:
        raise RuntimeError("evaluation worker cache was not initialized")
    return evaluate_ordered_pair(_WORKER_CACHE, **task)


def run_target(
    target: str,
    seeds: int,
    max_pairs: int | None,
    animal: str,
    r1_sessions,
    r2_sessions,
    workers: int,
) -> pd.DataFrame:
    print(f"\n########## {target} ##########", flush=True)
    pairs = list(product(r1_sessions, r2_sessions))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    sessions = list(dict.fromkeys(
        session
        for r1_session, r2_session in pairs
        for session in (r1_session, r2_session)
    ))
    cache = {
        session: load_session(session, target, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }

    tasks = []
    for seed in range(seeds):
        for pair_index, (r1_session, r2_session) in enumerate(pairs):
            tasks.extend([
                {
                    "train_session": r1_session,
                    "test_session": r2_session,
                    "direction": "R1->R2",
                    "target": target,
                    "seed": seed,
                    "pair_index": pair_index,
                    "direction_index": 0,
                },
                {
                    "train_session": r2_session,
                    "test_session": r1_session,
                    "direction": "R2->R1",
                    "target": target,
                    "seed": seed,
                    "pair_index": pair_index,
                    "direction_index": 1,
                },
            ])

    worker_count = min(workers, len(tasks))
    print(
        f"  evaluating {len(tasks)} ordered pair-seeds with {worker_count} workers",
        flush=True,
    )
    if worker_count == 1:
        rows = [evaluate_ordered_pair(cache, **task) for task in tasks]
    else:
        rows = []
        progress_step = max(1, len(tasks) // 10)
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_evaluation_worker,
            initargs=(cache,),
        ) as executor:
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if completed % progress_step == 0 or completed == len(tasks):
                    print(f"  completed {completed}/{len(tasks)} evaluations", flush=True)

    result = pd.DataFrame(rows)
    result.insert(0, "animal", animal)
    return result.sort_values(
        ["target", "seed", "pair_index", "direction_index"],
        ignore_index=True,
    )


def seed_summary(pair_rows: pd.DataFrame) -> pd.DataFrame:
    direction = pair_rows.groupby(["animal", "target", "seed", "direction"]).agg(
        baseline_corr=("baseline_corr", "mean"),
        matched_corr=("matched_corr", "mean"),
        train_total_var_before=("train_total_var_before", "mean"),
        train_total_var_after=("train_total_var_after", "mean"),
        test_total_var=("test_total_var", "mean"),
        added_var_per_dim=("added_var_per_dim", "mean"),
        needs_noise_n=("needs_noise", "sum"),
        n_pairs=("needs_noise", "size"),
        variance_error_after=("variance_error_after", "mean"),
    ).reset_index()

    rows = []
    for (animal, target, seed), group in direction.groupby(["animal", "target", "seed"]):
        by_direction = group.set_index("direction")
        fwd = by_direction.loc["R1->R2"]
        rev = by_direction.loc["R2->R1"]
        rows.append({
            "animal": animal,
            "target": target,
            "seed": seed,
            "fwd_baseline": fwd.baseline_corr,
            "fwd_matched": fwd.matched_corr,
            "rev_baseline": rev.baseline_corr,
            "rev_matched": rev.matched_corr,
            "baseline_gap": rev.baseline_corr - fwd.baseline_corr,
            "main_matched_gap": rev.baseline_corr - fwd.matched_corr,
            "symmetric_matched_gap": rev.matched_corr - fwd.matched_corr,
            "fwd_delta": fwd.matched_corr - fwd.baseline_corr,
            "rev_delta": rev.matched_corr - rev.baseline_corr,
            "fwd_train_total_var_before": fwd.train_total_var_before,
            "fwd_train_total_var_after": fwd.train_total_var_after,
            "fwd_test_total_var": fwd.test_total_var,
            "fwd_added_var_per_dim": fwd.added_var_per_dim,
            "fwd_needs_noise_n": fwd.needs_noise_n,
            "fwd_n_pairs": fwd.n_pairs,
            "fwd_variance_error_after": fwd.variance_error_after,
            "rev_train_total_var_before": rev.train_total_var_before,
            "rev_train_total_var_after": rev.train_total_var_after,
            "rev_test_total_var": rev.test_total_var,
            "rev_added_var_per_dim": rev.added_var_per_dim,
            "rev_needs_noise_n": rev.needs_noise_n,
            "rev_n_pairs": rev.n_pairs,
            "rev_variance_error_after": rev.variance_error_after,
        })
    out = pd.DataFrame(rows)
    out["main_gap_retained"] = out["main_matched_gap"] / out["baseline_gap"]
    out["symmetric_gap_retained"] = out["symmetric_matched_gap"] / out["baseline_gap"]
    return out


def final_summary(seed_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fwd_baseline",
        "fwd_matched",
        "rev_baseline",
        "rev_matched",
        "baseline_gap",
        "main_matched_gap",
        "symmetric_matched_gap",
        "main_gap_retained",
        "symmetric_gap_retained",
        "fwd_delta",
        "rev_delta",
        "fwd_train_total_var_before",
        "fwd_train_total_var_after",
        "fwd_test_total_var",
        "fwd_added_var_per_dim",
        "fwd_needs_noise_n",
        "fwd_n_pairs",
        "fwd_variance_error_after",
        "rev_needs_noise_n",
    ]
    frames = []
    for (animal, target), group in seed_rows.groupby(["animal", "target"]):
        row = {"animal": animal, "target": target, "n_seeds": len(group)}
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        frames.append(row)
    return pd.DataFrame(frames)


def mean_sd(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def plot_results(seed_rows: pd.DataFrame, summary: pd.DataFrame, output: Path, animal: str):
    targets = list(seed_rows["target"].unique())
    fig, axes = plt.subplots(
        len(targets),
        3,
        figsize=(14, 4.2 * len(targets)),
        squeeze=False,
    )
    colors = {
        "R1 train": "#e74c3c",
        "R1 train + noise": "#c0392b",
        "paired R2": "#3498db",
        "gap": "#555555",
    }
    for row_index, target in enumerate(targets):
        target_rows = seed_rows[seed_rows.target == target]
        target_summary = summary[summary.target == target].iloc[0]

        ax = axes[row_index, 0]
        var_labels = ["R1 train", "R1 train + noise", "paired R2"]
        var_values = [
            target_rows["fwd_train_total_var_before"],
            target_rows["fwd_train_total_var_after"],
            target_rows["fwd_test_total_var"],
        ]
        means = [mean_sd(values)[0] for values in var_values]
        sds = [mean_sd(values)[1] for values in var_values]
        x = np.arange(len(var_labels))
        ax.bar(x, means, yerr=sds, capsize=4, color=[colors[v] for v in var_labels])
        ax.set_xticks(x, var_labels, rotation=20, ha="right")
        ax.set_ylabel("total variance (trace)")
        ax.set_title(
            f"{target}: variance check\n"
            f"R1->R2 noised {target_summary.fwd_needs_noise_n:.1f}/"
            f"{target_summary.fwd_n_pairs:.0f} pairs on average"
        )
        ax.grid(axis="y", alpha=0.25)

        ax = axes[row_index, 1]
        decode_labels = ["R1->R2\nbaseline", "R1->R2\nmatched", "R2->R1\nbaseline"]
        decode_values = [
            target_rows["fwd_baseline"],
            target_rows["fwd_matched"],
            target_rows["rev_baseline"],
        ]
        means = [mean_sd(values)[0] for values in decode_values]
        sds = [mean_sd(values)[1] for values in decode_values]
        x = np.arange(len(decode_labels))
        ax.bar(x, means, yerr=sds, capsize=4, color=["#e74c3c", "#c0392b", "#3498db"])
        ax.set_xticks(x, decode_labels)
        ax.set_ylabel("cross-day decode corr")
        ax.set_title("Decode after total-variance match")
        ax.grid(axis="y", alpha=0.25)

        ax = axes[row_index, 2]
        gap_labels = ["baseline", "matched"]
        gap_values = [target_rows["baseline_gap"], target_rows["main_matched_gap"]]
        means = [mean_sd(values)[0] for values in gap_values]
        sds = [mean_sd(values)[1] for values in gap_values]
        x = np.arange(len(gap_labels))
        ax.bar(x, means, yerr=sds, capsize=4, color=["#777777", "#444444"])
        retained = target_summary.main_gap_retained * 100
        ax.set_xticks(x, gap_labels)
        ax.set_ylabel("R2->R1 minus R1->R2")
        ax.set_title(f"Asymmetry retained: {retained:.1f}%")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"{animal} H0 total-variance control: increase-only isotropic noise in the training day",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal",
        type=str.upper,
        choices=sorted(ANIMAL_SESSIONS),
        default="TS",
    )
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--skip-figure", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    r1_sessions, r2_sessions = ANIMAL_SESSIONS[args.animal]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = [
        run_target(
            target,
            args.seeds,
            args.max_pairs,
            args.animal,
            r1_sessions,
            r2_sessions,
            args.workers,
        )
        for target in args.targets
    ]
    pair_rows = pd.concat(all_rows, ignore_index=True)
    suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
    if args.max_pairs is not None or args.seeds != N_SEEDS:
        suffix += "_smoke"
    pair_csv = PAIR_CSV.with_name(PAIR_CSV.stem + suffix + PAIR_CSV.suffix)
    seed_csv = SEED_CSV.with_name(SEED_CSV.stem + suffix + SEED_CSV.suffix)
    summary_csv = SUMMARY_CSV.with_name(SUMMARY_CSV.stem + suffix + SUMMARY_CSV.suffix)
    fig_path = FIG.with_name(FIG.stem + suffix + FIG.suffix)

    pair_rows.to_csv(pair_csv, index=False)
    seed_rows = seed_summary(pair_rows)
    seed_rows.to_csv(seed_csv, index=False)
    summary = final_summary(seed_rows)
    summary.to_csv(summary_csv, index=False)

    print("\nSummary:")
    display_cols = [
        "target",
        "fwd_baseline",
        "fwd_matched",
        "rev_baseline",
        "baseline_gap",
        "main_matched_gap",
        "main_gap_retained",
        "fwd_needs_noise_n",
        "fwd_n_pairs",
    ]
    print(summary[display_cols].round(4).to_string(index=False))

    if not args.skip_figure:
        plot_results(seed_rows, summary, fig_path, args.animal)
        print(f"\nsaved {fig_path}")
    print(f"saved {pair_csv}")
    print(f"saved {seed_csv}")
    print(f"saved {summary_csv}")


if __name__ == "__main__":
    freeze_support()
    main()
