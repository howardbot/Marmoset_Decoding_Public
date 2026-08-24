"""Aggregate cross-fitted ridge/Kalman control-space geometry by session and pair."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[2]
RESULT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "kalman_control_space"


def load_shards(prefix: str) -> pd.DataFrame:
    paths = sorted(RESULT_DIR.glob(f"{prefix}_job_*_of_*.csv"))
    paths = [path for path in paths if "smoke" not in path.name]
    if not paths:
        raise FileNotFoundError(f"no completed {prefix} shards in {RESULT_DIR}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def numeric_means(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).columns.difference(
        ["repeat", "fold"]
    )
    return frame.groupby(groups, as_index=False)[list(numeric)].mean()


def main():
    within = load_shards("within")
    crossday = load_shards("crossday")

    n_sessions = within["session"].nunique()
    n_pairs = crossday[["category", "first_session", "second_session"]].drop_duplicates()
    if n_sessions != 17 or len(within) != 17 * 5 * 5:
        raise RuntimeError(
            f"incomplete within analysis: {n_sessions} sessions, {len(within)} rows"
        )
    if len(n_pairs) != 136 or len(crossday) != 136 * 5 * 5:
        raise RuntimeError(
            f"incomplete crossday analysis: {len(n_pairs)} pairs, {len(crossday)} rows"
        )

    within_session = numeric_means(within, ["target", "epoch", "session"])
    within_epoch = numeric_means(within_session, ["target", "epoch"])
    crossday_pair = numeric_means(
        crossday, ["target", "category", "first_session", "second_session"]
    )
    crossday_category = numeric_means(crossday_pair, ["target", "category"])

    within_session.to_csv(RESULT_DIR / "within_session_means.csv", index=False)
    within_epoch.to_csv(RESULT_DIR / "within_epoch_summary.csv", index=False)
    crossday_pair.to_csv(RESULT_DIR / "crossday_pair_means.csv", index=False)
    crossday_category.to_csv(RESULT_DIR / "crossday_category_summary.csv", index=False)

    print("\nWITHIN-SESSION, SESSION-WEIGHTED")
    print(within_epoch[[
        "epoch",
        "ridge_potent_fraction",
        "kalman_potent_fraction",
        "ridge_top_pc_mean_angle_deg",
        "kalman_top_pc_mean_angle_deg",
        "ridge_kalman_mean_angle_deg",
    ]].round(3).to_string(index=False))
    print("\nCROSS-DAY, PAIR-WEIGHTED")
    print(crossday_category[[
        "category",
        "ridge_crossday_mean_angle_deg",
        "kalman_crossday_mean_angle_deg",
        "first_ridge_kalman_mean_angle_deg",
        "second_ridge_kalman_mean_angle_deg",
    ]].round(3).to_string(index=False))
    print(
        "\nMAX RIDGE-KALMAN ANGLE (deg): "
        f"within={within.ridge_kalman_max_angle_deg.max():.6f}, "
        f"crossday={max(crossday.first_ridge_kalman_max_angle_deg.max(), crossday.second_ridge_kalman_max_angle_deg.max()):.6f}"
    )
    print(f"\nsaved summaries to {RESULT_DIR}")


if __name__ == "__main__":
    main()
