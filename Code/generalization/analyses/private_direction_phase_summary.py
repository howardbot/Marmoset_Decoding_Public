"""Aggregate private-direction encoding, error, phase, and trial localization."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))

from private_readout_crossfit_summary import summarize_pair_metrics

REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "manifold_geometry" / "private_direction_localization"
OUT_SUMMARY = IN_DIR / "localization_summary.csv"
OUT_DAY = IN_DIR / "localization_by_r2_session.csv"
OUT_TRIANGLE = IN_DIR / "triangle_cells.csv"
OUT_TRIANGLE_SUMMARY = IN_DIR / "triangle_cell_summary.csv"
OUT_TRIAL_CORR = IN_DIR / "trial_correlations.csv"
PHASE_WINDOWS = {"early": range(0, 10), "middle": range(10, 20), "late": range(20, 30)}


def read_jobs(prefix):
    files = [IN_DIR / f"{prefix}_job_{index}_of_8.csv" for index in range(8)]
    if not all(path.exists() for path in files):
        missing = [str(path) for path in files if not path.exists()]
        raise FileNotFoundError(f"missing {prefix} jobs: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame.to_csv(IN_DIR / f"{prefix}_all.csv", index=False)
    return frame


def pair_means(frame, grouping, metrics, output_name):
    result = frame.groupby(
        grouping + ["r1_session", "r2_session"], as_index=False
    )[metrics].mean()
    result.to_csv(IN_DIR / output_name, index=False)
    return result


def phase_cluster_tables(frame, grouping, metrics, prefix):
    by_day = frame.groupby(
        grouping + ["r2_session"], as_index=False
    )[metrics].mean()
    cluster = by_day.groupby(grouping, as_index=False)[metrics].mean()
    by_day.to_csv(IN_DIR / f"{prefix}_by_r2_session.csv", index=False)
    cluster.to_csv(IN_DIR / f"{prefix}_cluster_means.csv", index=False)
    return by_day, cluster


def add_phase_window(frame):
    frame = frame.copy()
    frame["phase_window"] = pd.cut(
        frame.phase_bin,
        bins=[-1, 9, 19, 29],
        labels=["early", "middle", "late"],
    ).astype(str)
    return frame


def trial_correlations(trials):
    aggregate_columns = [
        "success", "duration_s", "mean_speed", "peak_speed",
        "endpoint_x", "endpoint_y", "endpoint_z", "endpoint_distance",
        "full_activity_energy", "private_energy", "random_private_energy",
        "position_nmse_rescue", "position_corr_rescue",
        "full_position_nmse", "full_position_corr",
    ]
    trial_mean = trials.groupby(
        ["r1_session", "r2_session", "direction", "target_trial"],
        as_index=False,
    )[aggregate_columns].mean()
    trial_mean.to_csv(IN_DIR / "trials_averaged.csv", index=False)
    predictors = [
        "private_energy", "random_private_energy", "full_activity_energy",
        "duration_s", "mean_speed", "peak_speed",
        "endpoint_distance", "endpoint_x", "endpoint_y", "endpoint_z",
        "success",
    ]
    rows = []
    for keys, frame in trial_mean.groupby(
        ["r1_session", "r2_session", "direction"], sort=False
    ):
        for predictor in predictors:
            rows.append({
                "r1_session": keys[0],
                "r2_session": keys[1],
                "direction": keys[2],
                "predictor": predictor,
                "rho_nmse_rescue": frame[predictor].corr(
                    frame.position_nmse_rescue, method="spearman"
                ),
                "rho_corr_rescue": frame[predictor].corr(
                    frame.position_corr_rescue, method="spearman"
                ),
                "rho_full_nmse": frame[predictor].corr(
                    frame.full_position_nmse, method="spearman"
                ),
                "rho_full_corr": frame[predictor].corr(
                    frame.full_position_corr, method="spearman"
                ),
            })
    result = pd.DataFrame(rows)
    keys = ["r1_session", "r2_session", "direction"]
    metrics = ["rho_nmse_rescue", "rho_corr_rescue", "rho_full_nmse", "rho_full_corr"]
    private = result[result.predictor == "private_energy"].set_index(keys)[metrics]
    contrasts = []
    for control in ("random_private_energy", "full_activity_energy"):
        control_values = result[result.predictor == control].set_index(keys)[metrics]
        contrast = (private - control_values).reset_index()
        contrast["predictor"] = f"private_minus_{control}"
        contrasts.append(contrast)
    result = pd.concat([result, *contrasts], ignore_index=True)
    result.to_csv(OUT_TRIAL_CORR, index=False)
    return result


def triangle_table(encoding_phase_pair, error_phase_pair):
    encoding = add_phase_window(encoding_phase_pair).groupby(
        ["feature", "phase_window", "r1_session", "r2_session"],
        as_index=False,
    )[["private_corr", "random_private_corr", "private_excess_random"]].mean()
    error = add_phase_window(error_phase_pair).groupby(
        ["direction", "feature", "phase_window", "r1_session", "r2_session"],
        as_index=False,
    )[["full_nmse", "nmse_rescue", "corr_rescue"]].mean()
    wide = error.pivot(
        index=["feature", "phase_window", "r1_session", "r2_session"],
        columns="direction",
        values=["full_nmse", "nmse_rescue", "corr_rescue"],
    ).reset_index()
    wide.columns = [
        column if isinstance(column, str) else (
            column[0] if not column[1] else f"{column[0]}_{column[1]}"
        )
        for column in wide.columns
    ]
    triangle = encoding.merge(
        wide,
        on=["feature", "phase_window", "r1_session", "r2_session"],
        validate="one_to_one",
    )
    triangle["selective_nmse_rescue"] = (
        triangle.nmse_rescue_forward - triangle.nmse_rescue_reverse
    )
    triangle["selective_corr_rescue"] = (
        triangle.corr_rescue_forward - triangle.corr_rescue_reverse
    )
    by_day = triangle.groupby(
        ["feature", "phase_window", "r2_session"], as_index=False
    ).mean(numeric_only=True)
    cluster = by_day.groupby(
        ["feature", "phase_window"], as_index=False
    ).mean(numeric_only=True)
    cluster.to_csv(OUT_TRIANGLE, index=False)
    by_day.to_csv(IN_DIR / "triangle_cells_by_r2_session.csv", index=False)
    triangle.to_csv(IN_DIR / "triangle_pair_means.csv", index=False)
    return triangle, by_day, cluster


def main():
    encoding_overall = read_jobs("encoding_overall")
    encoding_phase = read_jobs("encoding_phase")
    error_overall = read_jobs("error_overall")
    error_phase = read_jobs("error_phase")
    trials = read_jobs("trials")

    encoding_metrics = [
        "private_corr", "random_private_corr", "private_excess_random",
        "shared_corr", "random_shared_corr", "shared_excess_random", "full_corr",
        "rank_shared", "rank_private_r1", "rank_private_r2",
    ]
    error_metrics = [
        "full_corr", "ablated_corr", "corr_rescue",
        "full_nmse", "ablated_nmse", "nmse_rescue",
    ]
    encoding_pair = pair_means(
        encoding_overall,
        ["feature"],
        encoding_metrics,
        "encoding_overall_pair_means.csv",
    )
    encoding_phase_pair = pair_means(
        encoding_phase,
        ["feature", "phase_bin", "phase_fraction"],
        encoding_metrics[:7],
        "encoding_phase_pair_means.csv",
    )
    error_pair = pair_means(
        error_overall,
        ["direction", "feature"],
        error_metrics,
        "error_overall_pair_means.csv",
    )
    error_phase_pair = pair_means(
        error_phase,
        ["direction", "feature", "phase_bin", "phase_fraction"],
        error_metrics,
        "error_phase_pair_means.csv",
    )
    phase_cluster_tables(
        encoding_phase_pair,
        ["feature", "phase_bin", "phase_fraction"],
        ["private_corr", "random_private_corr", "private_excess_random", "shared_corr"],
        "encoding_phase",
    )
    phase_cluster_tables(
        error_phase_pair,
        ["direction", "feature", "phase_bin", "phase_fraction"],
        ["full_corr", "corr_rescue", "full_nmse", "nmse_rescue"],
        "error_phase",
    )
    triangle_pair, _, triangle_cluster = triangle_table(
        encoding_phase_pair, error_phase_pair
    )
    trial_pair = trial_correlations(trials)

    day_rows = []
    summary_rows = []
    days, summaries = summarize_pair_metrics(
        encoding_pair,
        ["feature"],
        encoding_metrics,
        "private_encoding",
    )
    day_rows.extend(days)
    summary_rows.extend(summaries)
    days, summaries = summarize_pair_metrics(
        error_pair,
        ["direction", "feature"],
        error_metrics,
        "private_ablation_error",
    )
    day_rows.extend(days)
    summary_rows.extend(summaries)
    days, summaries = summarize_pair_metrics(
        trial_pair,
        ["direction", "predictor"],
        ["rho_nmse_rescue", "rho_corr_rescue", "rho_full_nmse", "rho_full_corr"],
        "trial_stratification",
    )
    day_rows.extend(days)
    summary_rows.extend(summaries)

    triangle_metrics = [
        "private_excess_random", "full_nmse_forward",
        "nmse_rescue_forward", "nmse_rescue_reverse",
        "selective_nmse_rescue", "corr_rescue_forward",
        "corr_rescue_reverse", "selective_corr_rescue",
    ]
    triangle_day_rows, triangle_summary_rows = summarize_pair_metrics(
        triangle_pair,
        ["feature", "phase_window"],
        triangle_metrics,
        "triangle_cells",
    )
    pd.DataFrame(triangle_summary_rows).to_csv(OUT_TRIANGLE_SUMMARY, index=False)
    pd.DataFrame(triangle_day_rows).to_csv(
        IN_DIR / "triangle_cell_by_r2_session.csv", index=False
    )

    top_encoding = triangle_cluster.sort_values(
        "private_excess_random", ascending=False
    ).iloc[0]
    selected = triangle_pair[
        (triangle_pair.feature == top_encoding.feature)
        & (triangle_pair.phase_window == top_encoding.phase_window)
    ]
    days, summaries = summarize_pair_metrics(
        selected,
        [],
        triangle_metrics,
        "selected_triangle_cell",
    )
    for row in days:
        row.update({
            "selected_feature": top_encoding.feature,
            "selected_phase_window": top_encoding.phase_window,
        })
    for row in summaries:
        row.update({
            "selected_feature": top_encoding.feature,
            "selected_phase_window": top_encoding.phase_window,
        })
    day_rows.extend(days)
    summary_rows.extend(summaries)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(day_rows).to_csv(OUT_DAY, index=False)

    print("Top overall private encoding beyond random:")
    print(
        summary[
            (summary.analysis == "private_encoding")
            & (summary.metric == "private_excess_random")
        ][["feature", "cluster_mean", "hier_boot_lo", "hier_boot_hi",
           "min_r2_day", "max_r2_day"]]
        .sort_values("cluster_mean", ascending=False)
        .round(4)
        .to_string(index=False)
    )
    print(
        f"\nSelected encoding cell: {top_encoding.feature} / "
        f"{top_encoding.phase_window}"
    )
    print(
        summary[summary.analysis == "selected_triangle_cell"]
        [["metric", "cluster_mean", "hier_boot_lo", "hier_boot_hi",
          "min_r2_day", "max_r2_day", "all_r2_days_positive"]]
        .round(4)
        .to_string(index=False)
    )
    cell_means = triangle_cluster[[
        "private_excess_random", "full_nmse_forward",
        "selective_nmse_rescue", "selective_corr_rescue",
    ]]
    print("\nAcross 21 feature x phase-window cells (Spearman):")
    print(cell_means.corr(method="spearman").round(3).to_string())
    print(f"\nsaved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
