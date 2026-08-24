"""Aggregate repeated private-readout cross-fit shards at the R2-session level."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "private_readout_crossfit"
OUT_SUBSPACE = IN_DIR / "subspace_all.csv"
OUT_KALMAN_SUBSPACE = IN_DIR / "kalman_subspace_all.csv"
OUT_MAPS = IN_DIR / "maps_all.csv"
OUT_KALMAN = IN_DIR / "kalman_maps_all.csv"
OUT_PAIR = IN_DIR / "subspace_pair_means.csv"
OUT_KALMAN_PAIR = IN_DIR / "kalman_subspace_pair_means.csv"
OUT_DAY = IN_DIR / "mechanism_by_r2_session.csv"
OUT_SUMMARY = IN_DIR / "mechanism_summary.csv"
N_BOOT = 20_000
SEED = 20260713


def add_subspace_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["gap_full"] = frame.rev_full - frame.fwd_full
    frame["selective_shared"] = (
        (frame.fwd_shared - frame.fwd_full)
        - (frame.rev_shared - frame.rev_full)
    )
    frame["selective_random_shared"] = (
        (frame.fwd_random_shared_mean - frame.fwd_full)
        - (frame.rev_random_shared_mean - frame.rev_full)
    )
    frame["shared_excess_over_random"] = (
        frame.selective_shared - frame.selective_random_shared
    )
    frame["selective_remove_r1_private"] = (
        (frame.fwd_minus_r1_private - frame.fwd_full)
        - (frame.rev_minus_r1_private - frame.rev_full)
    )
    frame["selective_random_ablate_r1"] = (
        (frame.fwd_random_ablate_a_mean - frame.fwd_full)
        - (frame.rev_random_ablate_a_mean - frame.rev_full)
    )
    frame["r1_private_excess_over_random"] = (
        frame.selective_remove_r1_private - frame.selective_random_ablate_r1
    )
    frame["selective_remove_r2_private"] = (
        (frame.fwd_minus_r2_private - frame.fwd_full)
        - (frame.rev_minus_r2_private - frame.rev_full)
    )
    frame["selective_random_ablate_r2"] = (
        (frame.fwd_random_ablate_b_mean - frame.fwd_full)
        - (frame.rev_random_ablate_b_mean - frame.rev_full)
    )
    frame["r2_private_excess_over_random"] = (
        frame.selective_remove_r2_private - frame.selective_random_ablate_r2
    )
    return frame


def add_map_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["raw_directional_gap"] = frame.rev_cross_map - frame.fwd_cross_map
    frame["target_ceiling_gap"] = frame.own_r1_map - frame.own_r2_map
    frame["transfer_penalty_asymmetry"] = frame.fwd_map_loss - frame.rev_map_loss
    frame["decomposition_error"] = (
        frame.raw_directional_gap
        - frame.target_ceiling_gap
        - frame.transfer_penalty_asymmetry
    )
    return frame


def hierarchical_interval(pair_frame: pd.DataFrame, metric: str, rng) -> tuple[float, float]:
    days = [
        day for day in sorted(pair_frame.r2_session.unique())
        if pair_frame.loc[pair_frame.r2_session == day, metric].notna().any()
    ]
    if not days:
        return np.nan, np.nan
    values_by_day = [
        pair_frame.loc[pair_frame.r2_session == day, metric].dropna().to_numpy()
        for day in days
    ]
    sampled_days = rng.integers(0, len(days), size=(N_BOOT, len(days)))
    draws = np.zeros(N_BOOT)
    for slot in range(len(days)):
        for day_index, values in enumerate(values_by_day):
            selected = sampled_days[:, slot] == day_index
            n_selected = int(selected.sum())
            if n_selected == 0:
                continue
            pair_indices = rng.integers(
                0, len(values), size=(n_selected, len(values))
            )
            draws[selected] += values[pair_indices].mean(axis=1)
    draws /= len(days)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize_pair_metrics(pair_frame, grouping, metrics, analysis):
    rng = np.random.default_rng(SEED)
    day_rows = []
    summary_rows = []
    grouped = [((), pair_frame)] if not grouping else pair_frame.groupby(grouping)
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = dict(zip(grouping, keys))
        by_day = frame.groupby("r2_session")[metrics].mean().reset_index()
        for _, row in by_day.iterrows():
            day_rows.append({"analysis": analysis, **labels, **row.to_dict()})
        for metric in metrics:
            lo, hi = hierarchical_interval(frame, metric, rng)
            day_values = by_day[metric].dropna()
            if day_values.empty:
                summary_rows.append({
                    "analysis": analysis, **labels, "metric": metric,
                    "cluster_mean": np.nan, "hier_boot_lo": np.nan,
                    "hier_boot_hi": np.nan, "min_r2_day": np.nan,
                    "max_r2_day": np.nan, "all_r2_days_positive": False,
                    "n_r2_days": 0,
                    "interpretation": "metric unavailable at this threshold",
                })
                continue
            summary_rows.append({
                "analysis": analysis,
                **labels,
                "metric": metric,
                "cluster_mean": float(day_values.mean()),
                "hier_boot_lo": lo,
                "hier_boot_hi": hi,
                "min_r2_day": float(day_values.min()),
                "max_r2_day": float(day_values.max()),
                "all_r2_days_positive": bool((day_values > 0).all()),
                "n_r2_days": int(len(day_values)),
                "interpretation": "descriptive one-animal sensitivity interval",
            })
    return day_rows, summary_rows


def main():
    subspace = pd.concat(
        [pd.read_csv(IN_DIR / f"subspace_shard_{index}.csv") for index in range(3)],
        ignore_index=True,
    )
    maps = pd.concat(
        [pd.read_csv(IN_DIR / f"maps_shard_{index}.csv") for index in range(3)],
        ignore_index=True,
    )
    kalman_maps = pd.concat(
        [pd.read_csv(IN_DIR / f"kalman_maps_shard_{index}.csv") for index in range(3)],
        ignore_index=True,
    )
    job_files = [
        IN_DIR / f"kalman_subspace_job_{index}_of_8.csv" for index in range(8)
    ]
    kalman_subspace_files = job_files if all(path.exists() for path in job_files) else [
        IN_DIR / f"kalman_subspace_shard_{index}.csv" for index in range(3)
    ]
    kalman_subspace = pd.concat(
        [pd.read_csv(path) for path in kalman_subspace_files], ignore_index=True
    )
    subspace = add_subspace_contrasts(subspace)
    kalman_subspace = add_subspace_contrasts(kalman_subspace)
    maps = add_map_decomposition(maps)
    kalman_maps = add_map_decomposition(kalman_maps)
    kalman_maps["weight_cosine"] = np.nan
    subspace.to_csv(OUT_SUBSPACE, index=False)
    kalman_subspace.to_csv(OUT_KALMAN_SUBSPACE, index=False)
    maps.to_csv(OUT_MAPS, index=False)
    kalman_maps.to_csv(OUT_KALMAN, index=False)

    subspace_metrics = [
        "gap_full",
        "selective_shared",
        "shared_excess_over_random",
        "selective_remove_r1_private",
        "r1_private_excess_over_random",
        "selective_remove_r2_private",
        "r2_private_excess_over_random",
        "rank_shared",
        "rank_private_r1",
        "rank_private_r2",
    ]
    subspace_pair = subspace.groupby(
        ["cosine_threshold", "r1_session", "r2_session"], as_index=False
    )[subspace_metrics].mean()
    subspace_pair.to_csv(OUT_PAIR, index=False)
    kalman_subspace_pair = kalman_subspace.groupby(
        ["cosine_threshold", "r1_session", "r2_session"], as_index=False
    )[subspace_metrics].mean()
    kalman_subspace_pair.to_csv(OUT_KALMAN_PAIR, index=False)
    map_metrics = [
        "raw_directional_gap",
        "target_ceiling_gap",
        "transfer_penalty_asymmetry",
        "fwd_map_loss",
        "rev_map_loss",
        "weight_cosine",
        "decomposition_error",
    ]
    map_pair = maps.groupby(["r1_session", "r2_session"], as_index=False)[
        map_metrics
    ].mean()
    kalman_pair = kalman_maps.groupby(
        ["r1_session", "r2_session"], as_index=False
    )[map_metrics].mean()

    day_rows, summary_rows = summarize_pair_metrics(
        subspace_pair, ["cosine_threshold"], subspace_metrics,
        "subspace_intervention"
    )
    kalman_subspace_days, kalman_subspace_summary = summarize_pair_metrics(
        kalman_subspace_pair, ["cosine_threshold"], subspace_metrics,
        "kalman_subspace_intervention"
    )
    map_days, map_summary = summarize_pair_metrics(
        map_pair, [], map_metrics, "direct_map_decomposition"
    )
    kalman_days, kalman_summary = summarize_pair_metrics(
        kalman_pair, [], map_metrics, "kalman_map_decomposition"
    )
    day_rows.extend(map_days)
    summary_rows.extend(map_summary)
    day_rows.extend(kalman_subspace_days)
    summary_rows.extend(kalman_subspace_summary)
    day_rows.extend(kalman_days)
    summary_rows.extend(kalman_summary)
    days = pd.DataFrame(day_rows)
    summary = pd.DataFrame(summary_rows)
    days.to_csv(OUT_DAY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    headline = summary[
        ((summary.analysis.isin([
             "subspace_intervention", "kalman_subspace_intervention"
         ])) &
         (summary.cosine_threshold == 0.5) &
         summary.metric.isin([
             "gap_full", "selective_shared", "shared_excess_over_random",
             "selective_remove_r1_private", "r1_private_excess_over_random",
             "selective_remove_r2_private",
         ]))
        | ((summary.analysis.isin([
               "direct_map_decomposition", "kalman_map_decomposition"
           ])) &
           summary.metric.isin(map_metrics[:-1]))
    ]
    print(headline.round(4).to_string(index=False))
    print(f"\nsaved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
