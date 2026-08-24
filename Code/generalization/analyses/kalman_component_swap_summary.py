"""Aggregate repeated Kalman component swaps at the R2-session level."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from kalman_components import COMPONENT_NAMES, shapley_values
from private_readout_crossfit_summary import summarize_pair_metrics

REPO = _THIS.parents[2]
IN_DIR = REPO / "Results" / "manifold_geometry" / "kalman_component_swap"
REFERENCE = (
    REPO / "Results" / "manifold_geometry" / "private_readout_crossfit"
    / "kalman_maps_all.csv"
)
OUT_ALL = IN_DIR / "component_swap_all.csv"
OUT_PAIR = IN_DIR / "component_swap_pair_means.csv"
OUT_SHAPLEY = IN_DIR / "component_shapley_pair_means.csv"
OUT_DECOMPOSITION = IN_DIR / "component_decomposition_pair_means.csv"
OUT_KINEMATIC = IN_DIR / "kinematic_transition_all.csv"
OUT_KINEMATIC_PAIR = IN_DIR / "kinematic_transition_pair_means.csv"
OUT_DAY = IN_DIR / "component_swap_by_r2_session.csv"
OUT_SUMMARY = IN_DIR / "component_swap_summary.csv"
SPLIT_KEYS = [
    "transition_mode", "r1_session", "r2_session", "repeat", "fold"
]


def add_rescues(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame.target_mask == 0][
        SPLIT_KEYS + ["fwd_score", "rev_score"]
    ].rename(columns={"fwd_score": "fwd_baseline", "rev_score": "rev_baseline"})
    frame = frame.merge(baseline, on=SPLIT_KEYS, validate="many_to_one")
    frame["fwd_rescue"] = frame.fwd_score - frame.fwd_baseline
    frame["rev_rescue"] = frame.rev_score - frame.rev_baseline
    frame["selective_rescue"] = frame.fwd_rescue - frame.rev_rescue
    frame["directional_gap"] = frame.rev_score - frame.fwd_score
    return frame


def split_shapley(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, split in frame.groupby(SPLIT_KEYS, sort=False):
        forward = dict(zip(split.target_mask, split.fwd_score))
        reverse = dict(zip(split.target_mask, split.rev_score))
        forward_values = shapley_values(forward)
        reverse_values = shapley_values(reverse)
        for component in COMPONENT_NAMES:
            rows.append({
                **dict(zip(SPLIT_KEYS, keys)),
                "component": component,
                "fwd_shapley": forward_values[component],
                "rev_shapley": reverse_values[component],
                "selective_shapley": (
                    forward_values[component] - reverse_values[component]
                ),
            })
    result = pd.DataFrame(rows)
    sums = result.groupby(SPLIT_KEYS).selective_shapley.sum()
    full = frame[frame.target_mask == 15].set_index(SPLIT_KEYS).selective_rescue
    error = (sums - full).abs().max()
    if error > 1e-10:
        raise AssertionError(f"Shapley allocation error: {error}")
    return result


def check_concatenated_baseline(frame: pd.DataFrame):
    reference = pd.read_csv(REFERENCE)
    baseline = frame[
        (frame.transition_mode == "concatenated") & (frame.target_mask == 0)
    ]
    merged = baseline.merge(
        reference,
        on=["r1_session", "r2_session", "repeat", "fold"],
        validate="one_to_one",
    )
    forward_error = (merged.fwd_score - merged.fwd_cross_map).abs().max()
    reverse_error = (merged.rev_score - merged.rev_cross_map).abs().max()
    if max(forward_error, reverse_error) > 1e-10:
        raise AssertionError(
            f"source-decoder parity failed: forward={forward_error}, reverse={reverse_error}"
        )
    return float(forward_error), float(reverse_error)


def main():
    files = [IN_DIR / f"component_swap_job_{index}_of_8.csv" for index in range(8)]
    if not all(path.exists() for path in files):
        missing = [str(path) for path in files if not path.exists()]
        raise FileNotFoundError(f"missing component-swap jobs: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame = add_rescues(frame)
    parity = check_concatenated_baseline(frame)
    frame.to_csv(OUT_ALL, index=False)

    condition_metrics = [
        "fwd_score", "rev_score", "directional_gap",
        "fwd_rescue", "rev_rescue", "selective_rescue",
    ]
    pair = frame.groupby(
        ["transition_mode", "target_mask", "target_components",
         "r1_session", "r2_session"],
        as_index=False,
    )[condition_metrics].mean()
    pair.to_csv(OUT_PAIR, index=False)

    split_components = split_shapley(frame)
    shapley_metrics = ["fwd_shapley", "rev_shapley", "selective_shapley"]
    shapley_pair = split_components.groupby(
        ["transition_mode", "component", "r1_session", "r2_session"],
        as_index=False,
    )[shapley_metrics].mean()
    shapley_pair.to_csv(OUT_SHAPLEY, index=False)

    endpoints = pair[pair.target_mask.isin([0, 15])].pivot(
        index=["transition_mode", "r1_session", "r2_session"],
        columns="target_mask",
        values=["directional_gap", "selective_rescue"],
    ).reset_index()
    endpoints.columns = [
        "transition_mode", "r1_session", "r2_session",
        "raw_directional_gap", "fully_adapted_gap",
        "baseline_zero", "full_selective_rescue",
    ]
    endpoints = endpoints.drop(columns="baseline_zero")
    endpoints.to_csv(OUT_DECOMPOSITION, index=False)

    day_rows = []
    summary_rows = []
    selected_masks = [1, 2, 4, 8, 3, 12, 15]
    intervention_days, intervention_summary = summarize_pair_metrics(
        pair[pair.target_mask.isin(selected_masks)],
        ["transition_mode", "target_mask", "target_components"],
        ["fwd_rescue", "rev_rescue", "selective_rescue"],
        "component_intervention",
    )
    shapley_days, shapley_summary = summarize_pair_metrics(
        shapley_pair,
        ["transition_mode", "component"],
        shapley_metrics,
        "component_shapley",
    )
    decomposition_days, decomposition_summary = summarize_pair_metrics(
        endpoints,
        ["transition_mode"],
        ["raw_directional_gap", "fully_adapted_gap", "full_selective_rescue"],
        "component_decomposition",
    )
    kinematic_files = [
        IN_DIR / f"kinematic_transition_job_{index}_of_8.csv"
        for index in range(8)
    ]
    kinematic = pd.concat(
        [pd.read_csv(path) for path in kinematic_files], ignore_index=True
    )
    kinematic.to_csv(OUT_KINEMATIC, index=False)
    kinematic_metrics = [
        "fwd_source_nmse", "fwd_adapted_nmse",
        "rev_source_nmse", "rev_adapted_nmse",
        "fwd_transition_penalty", "rev_transition_penalty",
        "directional_transition_penalty",
    ]
    kinematic_pair = kinematic.groupby(
        ["r1_session", "r2_session"], as_index=False
    )[kinematic_metrics].mean()
    kinematic_pair.to_csv(OUT_KINEMATIC_PAIR, index=False)
    kinematic_days, kinematic_summary = summarize_pair_metrics(
        kinematic_pair,
        [],
        kinematic_metrics,
        "kinematic_transition",
    )
    day_rows.extend(
        intervention_days + shapley_days + decomposition_days + kinematic_days
    )
    summary_rows.extend(
        intervention_summary + shapley_summary + decomposition_summary
        + kinematic_summary
    )
    pd.DataFrame(day_rows).to_csv(OUT_DAY, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    headline = summary[
        ((summary.analysis == "component_shapley") &
         (summary.metric == "selective_shapley"))
        | ((summary.analysis == "component_decomposition") &
           summary.metric.isin([
               "raw_directional_gap", "fully_adapted_gap",
               "full_selective_rescue",
           ]))
        | ((summary.analysis == "component_intervention") &
           summary.target_mask.isin([3, 12]) &
           (summary.metric == "selective_rescue"))
        | ((summary.analysis == "kinematic_transition") &
           summary.metric.isin([
               "fwd_transition_penalty", "rev_transition_penalty",
               "directional_transition_penalty",
           ]))
    ]
    print(headline.round(4).to_string(index=False))
    print(f"\nsource-decoder parity max errors: {parity}")
    print(f"saved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
