"""Aggregate the final private/axis/intercept decomposition of Kalman H."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

from h_observation_decomposition import shapley_values_named
from private_readout_crossfit_summary import summarize_pair_metrics


REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "h_observation_fine_swap"
REFERENCE = (
    REPO / "Results" / "workflows" / "manifold_geometry" / "kalman_component_swap"
    / "component_swap_all.csv"
)
SPLIT_KEYS = ["r1_session", "r2_session", "repeat", "fold"]
SCORE_METRICS = [
    "fwd_score", "rev_score", "directional_gap",
    "fwd_rescue", "rev_rescue", "selective_rescue",
]


def read_jobs(prefix):
    files = [IN_DIR / f"h_{prefix}_job_{index}_of_8.csv" for index in range(8)]
    if not all(path.exists() for path in files):
        missing = [str(path) for path in files if not path.exists()]
        raise FileNotFoundError(f"missing {prefix} jobs: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame.to_csv(IN_DIR / f"h_{prefix}_all.csv", index=False)
    return frame


def pivot_direction(frame, condition_keys):
    index = condition_keys + SPLIT_KEYS
    if frame[index + ["direction"]].duplicated().any():
        raise AssertionError(f"duplicate direction scores for {condition_keys}")
    wide = frame.pivot(index=index, columns="direction", values="score").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"forward": "fwd_score", "reverse": "rev_score"})
    if wide[["fwd_score", "rev_score"]].isna().any().any():
        raise AssertionError("direction pivot produced missing scores")
    wide["directional_gap"] = wide.rev_score - wide.fwd_score
    return wide


def add_rescues(frame, group_keys, baseline_selector):
    baseline = frame[baseline_selector(frame)][
        group_keys + SPLIT_KEYS + ["fwd_score", "rev_score"]
    ].rename(columns={"fwd_score": "fwd_baseline", "rev_score": "rev_baseline"})
    if baseline[group_keys + SPLIT_KEYS].duplicated().any():
        raise AssertionError("baseline selector is not unique")
    result = frame.merge(
        baseline, on=group_keys + SPLIT_KEYS, validate="many_to_one"
    )
    result["fwd_rescue"] = result.fwd_score - result.fwd_baseline
    result["rev_rescue"] = result.rev_score - result.rev_baseline
    result["selective_rescue"] = result.fwd_rescue - result.rev_rescue
    return result


def factorial_shapley(frame, group_keys, mask_column, names):
    rows = []
    grouping = group_keys + SPLIT_KEYS
    expected = set(range(2 ** len(names)))
    for keys, split in frame.groupby(grouping, sort=False):
        if set(split[mask_column]) != expected:
            raise AssertionError(f"incomplete factorial for {dict(zip(grouping, keys))}")
        forward = dict(zip(split[mask_column], split.fwd_score))
        reverse = dict(zip(split[mask_column], split.rev_score))
        forward_values = shapley_values_named(forward, names)
        reverse_values = shapley_values_named(reverse, names)
        labels = dict(zip(grouping, keys))
        for name in names:
            rows.append({
                **labels,
                "component": name,
                "fwd_shapley": forward_values[name],
                "rev_shapley": reverse_values[name],
                "selective_shapley": (
                    forward_values[name] - reverse_values[name]
                ),
            })
    result = pd.DataFrame(rows)
    sums = result.groupby(grouping).selective_shapley.sum()
    endpoints = frame[frame[mask_column] == 2 ** len(names) - 1].set_index(
        grouping
    ).selective_rescue
    error = float((sums - endpoints).abs().max())
    if error > 1e-10:
        raise AssertionError(f"Shapley endpoint error: {error}")
    return result


def prepare_partition(frame):
    keys = [
        "observation_family", "q_context", "condition", "random_draw"
    ]
    wide = pivot_direction(frame, keys)
    wide = add_rescues(
        wide,
        ["observation_family", "q_context"],
        lambda values: values.condition == "source",
    )
    random_mean = wide[wide.condition == "random_delta"].groupby(
        ["observation_family", "q_context"] + SPLIT_KEYS,
        as_index=False,
    )[SCORE_METRICS].mean()
    random_mean["condition"] = "random_mean"
    random_mean["random_draw"] = -1
    analysis = pd.concat([
        wide[wide.condition != "random_delta"], random_mean
    ], ignore_index=True)

    private = analysis[analysis.condition == "private_delta"]
    random = analysis[analysis.condition == "random_mean"]
    adjusted = private.merge(
        random,
        on=["observation_family", "q_context"] + SPLIT_KEYS,
        suffixes=("_private", "_random"),
        validate="one_to_one",
    )
    for metric in SCORE_METRICS:
        adjusted[metric] = adjusted[f"{metric}_private"] - adjusted[f"{metric}_random"]
    adjusted["condition"] = "private_minus_random"
    adjusted = adjusted[
        ["observation_family", "q_context", "condition"]
        + SPLIT_KEYS + SCORE_METRICS
    ]

    factorial = analysis[analysis.condition.isin([
        "source", "private_delta", "rest_delta", "target_H"
    ])].copy()
    factorial["partition_mask"] = factorial.condition.map({
        "source": 0,
        "private_delta": 1,
        "rest_delta": 2,
        "target_H": 3,
    })
    shapley = factorial_shapley(
        factorial,
        ["observation_family", "q_context"],
        "partition_mask",
        ("private", "rest"),
    )
    return wide, analysis, adjusted, shapley


def prepare_axis(frame):
    wide = pivot_direction(
        frame,
        ["observation_family", "q_context", "delta_scope", "axis_mask"],
    )
    wide = add_rescues(
        wide,
        ["observation_family", "q_context", "delta_scope"],
        lambda values: values.axis_mask == 0,
    )
    shapley = factorial_shapley(
        wide,
        ["observation_family", "q_context", "delta_scope"],
        "axis_mask",
        ("x", "y", "z"),
    ).rename(columns={"component": "axis"})
    return wide, shapley


def prepare_affine(frame):
    wide = pivot_direction(
        frame,
        ["observation_family", "q_context", "affine_mask"],
    )
    wide = add_rescues(
        wide,
        ["observation_family", "q_context"],
        lambda values: values.affine_mask == 0,
    )
    shapley = factorial_shapley(
        wide,
        ["observation_family", "q_context"],
        "affine_mask",
        ("b", "H"),
    )
    return wide, shapley


def prepare_centering(frame):
    identity_error = float(frame.target_identity_error.max())
    if identity_error > 1e-10:
        raise AssertionError(f"affine-centering identity failed: {identity_error}")
    wide = pivot_direction(frame, ["q_context", "condition"])
    wide = add_rescues(
        wide,
        ["q_context"],
        lambda values: values.condition == "source_b",
    )
    target = wide[wide.condition == "target_b"]
    behaviour = wide[wide.condition == "behaviour_center"]
    contrast = target.merge(
        behaviour,
        on=["q_context"] + SPLIT_KEYS,
        suffixes=("_target", "_behaviour"),
        validate="one_to_one",
    )
    for metric in SCORE_METRICS:
        contrast[metric] = (
            contrast[f"{metric}_target"] - contrast[f"{metric}_behaviour"]
        )
    contrast["condition"] = "target_minus_behaviour"
    contrast = contrast[["q_context", "condition"] + SPLIT_KEYS + SCORE_METRICS]
    return wide, contrast, identity_error


def check_legacy_parity(partition):
    reference = pd.read_csv(REFERENCE)
    reference = reference[reference.transition_mode == "trial_aware"]
    mappings = {
        ("source", "source"): 0,
        ("source", "target_H"): 4,
        ("target", "source"): 8,
        ("target", "target_H"): 12,
    }
    errors = []
    for (q_context, condition), target_mask in mappings.items():
        current = partition[
            (partition.observation_family == "no_intercept")
            & (partition.q_context == q_context)
            & (partition.condition == condition)
            & (partition.random_draw == -1)
        ]
        expected = reference[reference.target_mask == target_mask]
        merged = current.merge(
            expected,
            on=SPLIT_KEYS,
            suffixes=("_current", "_reference"),
            validate="one_to_one",
        )
        errors.extend([
            float((merged.fwd_score_current - merged.fwd_score_reference).abs().max()),
            float((merged.rev_score_current - merged.rev_score_reference).abs().max()),
        ])
    maximum = max(errors)
    if maximum > 1e-10:
        raise AssertionError(f"legacy H/Q endpoint parity failed: {maximum}")
    return maximum


def check_internal_endpoints(partition, axis, affine):
    base_keys = ["observation_family", "q_context"] + SPLIT_KEYS
    comparisons = []
    comparisons.append((
        axis[(axis.delta_scope == "full") & (axis.axis_mask == 7)],
        partition[
            (partition.observation_family == "no_intercept")
            & (partition.condition == "target_H")
            & (partition.random_draw == -1)
        ],
        base_keys,
    ))
    comparisons.append((
        axis[(axis.delta_scope == "private") & (axis.axis_mask == 7)],
        partition[
            (partition.observation_family == "no_intercept")
            & (partition.condition == "private_delta")
            & (partition.random_draw == -1)
        ],
        base_keys,
    ))
    comparisons.append((
        affine[affine.affine_mask == 2],
        partition[
            (partition.observation_family == "affine")
            & (partition.condition == "target_H")
            & (partition.random_draw == -1)
        ],
        base_keys,
    ))
    errors = []
    for left, right, keys in comparisons:
        merged = left.merge(
            right,
            on=keys,
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        errors.extend([
            float((merged.fwd_score_left - merged.fwd_score_right).abs().max()),
            float((merged.rev_score_left - merged.rev_score_right).abs().max()),
        ])
    maximum = max(errors)
    if maximum > 1e-10:
        raise AssertionError(f"internal H endpoint parity failed: {maximum}")
    return maximum


def pair_means(frame, grouping, metrics, filename):
    result = frame.groupby(
        grouping + ["r1_session", "r2_session"], as_index=False
    )[metrics].mean()
    result.to_csv(IN_DIR / filename, index=False)
    return result


def main():
    partition_raw = read_jobs("partition")
    axis_raw = read_jobs("axis")
    affine_raw = read_jobs("affine")
    centering_raw = read_jobs("centering")

    partition, partition_analysis, adjusted, partition_shapley = prepare_partition(
        partition_raw
    )
    axis, axis_shapley = prepare_axis(axis_raw)
    affine, affine_shapley = prepare_affine(affine_raw)
    centering, centering_contrast, centering_identity_error = prepare_centering(
        centering_raw
    )
    legacy_error = check_legacy_parity(partition)
    internal_error = check_internal_endpoints(partition, axis, affine)

    partition_analysis.to_csv(IN_DIR / "h_partition_conditions.csv", index=False)
    adjusted.to_csv(IN_DIR / "h_partition_random_adjusted.csv", index=False)
    partition_shapley.to_csv(IN_DIR / "h_partition_shapley.csv", index=False)
    axis.to_csv(IN_DIR / "h_axis_conditions.csv", index=False)
    axis_shapley.to_csv(IN_DIR / "h_axis_shapley.csv", index=False)
    affine.to_csv(IN_DIR / "h_affine_conditions.csv", index=False)
    affine_shapley.to_csv(IN_DIR / "h_affine_shapley.csv", index=False)
    centering.to_csv(IN_DIR / "h_centering_conditions.csv", index=False)
    centering_contrast.to_csv(
        IN_DIR / "h_centering_contrast.csv", index=False
    )

    partition_pair = pair_means(
        partition_analysis,
        ["observation_family", "q_context", "condition"],
        SCORE_METRICS,
        "h_partition_pair_means.csv",
    )
    adjusted_pair = pair_means(
        adjusted,
        ["observation_family", "q_context", "condition"],
        SCORE_METRICS,
        "h_partition_random_adjusted_pair_means.csv",
    )
    shapley_metrics = ["fwd_shapley", "rev_shapley", "selective_shapley"]
    partition_shapley_pair = pair_means(
        partition_shapley,
        ["observation_family", "q_context", "component"],
        shapley_metrics,
        "h_partition_shapley_pair_means.csv",
    )
    axis_shapley_pair = pair_means(
        axis_shapley,
        ["observation_family", "q_context", "delta_scope", "axis"],
        shapley_metrics,
        "h_axis_shapley_pair_means.csv",
    )
    affine_shapley_pair = pair_means(
        affine_shapley,
        ["observation_family", "q_context", "component"],
        shapley_metrics,
        "h_affine_shapley_pair_means.csv",
    )
    centering_pair = pair_means(
        pd.concat([centering, centering_contrast], ignore_index=True),
        ["q_context", "condition"],
        SCORE_METRICS,
        "h_centering_pair_means.csv",
    )

    day_rows = []
    summary_rows = []
    analyses = [
        (
            partition_pair,
            ["observation_family", "q_context", "condition"],
            SCORE_METRICS,
            "h_partition_intervention",
        ),
        (
            adjusted_pair,
            ["observation_family", "q_context", "condition"],
            SCORE_METRICS,
            "h_partition_random_adjusted",
        ),
        (
            partition_shapley_pair,
            ["observation_family", "q_context", "component"],
            shapley_metrics,
            "h_partition_shapley",
        ),
        (
            axis_shapley_pair,
            ["observation_family", "q_context", "delta_scope", "axis"],
            shapley_metrics,
            "h_axis_shapley",
        ),
        (
            affine_shapley_pair,
            ["observation_family", "q_context", "component"],
            shapley_metrics,
            "h_affine_shapley",
        ),
        (
            centering_pair,
            ["q_context", "condition"],
            SCORE_METRICS,
            "h_intercept_centering",
        ),
    ]
    for frame, grouping, metrics, analysis in analyses:
        days, summaries = summarize_pair_metrics(
            frame, grouping, metrics, analysis
        )
        day_rows.extend(days)
        summary_rows.extend(summaries)
    pd.DataFrame(day_rows).to_csv(
        IN_DIR / "h_fine_by_r2_session.csv", index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(IN_DIR / "h_fine_summary.csv", index=False)

    headline = summary[
        (
            (summary.analysis == "h_partition_intervention")
            & summary.condition.isin([
                "private_delta", "rest_delta", "target_H", "random_mean"
            ])
            & (summary.metric == "selective_rescue")
        )
        | (
            (summary.analysis == "h_partition_random_adjusted")
            & (summary.metric == "selective_rescue")
        )
        | (
            (summary.analysis.isin([
                "h_partition_shapley", "h_axis_shapley", "h_affine_shapley"
            ]))
            & (summary.metric == "selective_shapley")
        )
        | (
            (summary.analysis == "h_intercept_centering")
            & summary.condition.isin([
                "behaviour_center", "target_b", "target_minus_behaviour"
            ])
            & (summary.metric == "selective_rescue")
        )
    ]
    print(headline.round(4).to_string(index=False))
    print(
        f"\nparity max errors: legacy={legacy_error:.3g}, "
        f"internal={internal_error:.3g}, "
        f"centering_identity={centering_identity_error:.3g}"
    )
    print(f"saved {IN_DIR / 'h_fine_summary.csv'}")


if __name__ == "__main__":
    main()
