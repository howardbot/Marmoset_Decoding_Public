"""Time-resolve the original TS interference gap after joint fixed-40 matching.

The selected trial IDs are the same neural+position joint subsets used by
``decode_variability_matched_crossday.py``.  Each of the 42 R1/R2 cells is
refit from its own 40/40 subset, then evaluated with the locked 30-ms
time-resolved position metric.  This supplies an equal-N comparison for the
forget-control time-localization analysis.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "marmoset_matplotlib")
)

import pandas as pd

THIS = Path(__file__).resolve()
GENERALIZATION = THIS.parents[1]
WHY = THIS.parent
for path in (GENERALIZATION, WHY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from big_sweep_phase2_crossday import SESSIONS_R1, SESSIONS_R2  # noqa: E402
from decode_variability_matched_crossday import (  # noqa: E402
    load_raw_session,
    selected_trials_for_pair,
    subset_cache,
)
from locked_position_time_resolved import (  # noqa: E402
    decode_with_predictions,
    plot_profiles,
    summarize_directional_gaps,
    summarize_windows,
    time_resolved_metrics,
)

REPO = THIS.parents[3]
RESULT_DIR = REPO / "Results" / "workflows" / "generalization"
FIGURE_DIR = RESULT_DIR / "figures"
SELECTIONS = (
    REPO
    / "Results"
    / "workflows"
    / "manifold_geometry"
    / "variability_match_all42_fixed40_tol10_trials.csv"
)
TARGET_MODE = "relative_position"
MATCH_MODE = "joint"
OUT_PROFILES = RESULT_DIR / "locked_position_time_resolved_ts_joint_fixed40.csv"
OUT_WINDOWS = RESULT_DIR / "locked_position_time_windows_ts_joint_fixed40.csv"
OUT_GAPS = RESULT_DIR / "locked_position_time_gap_ts_joint_fixed40.csv"
OUT_FIGURE = FIGURE_DIR / "fig_locked_position_time_resolved_ts_joint_fixed40.png"


def main() -> None:
    selections = pd.read_csv(SELECTIONS)
    selections = selections.loc[selections["match_mode"] == MATCH_MODE]
    sessions = tuple(SESSIONS_R1) + tuple(SESSIONS_R2)
    raw = {}
    for index, session in enumerate(sessions, start=1):
        print(f"[TS joint fixed40 cache {index}/{len(sessions)}] {session}", flush=True)
        raw[session] = load_raw_session(session, TARGET_MODE)

    rows = []
    pair_id = 0
    for r1_session in SESSIONS_R1:
        for r2_session in SESSIONS_R2:
            pair_id += 1
            r1_trials = selected_trials_for_pair(
                selections, pair_id, r1_session, MATCH_MODE
            )
            r2_trials = selected_trials_for_pair(
                selections, pair_id, r2_session, MATCH_MODE
            )
            if len(r1_trials) != 40 or len(r2_trials) != 40:
                raise ValueError(
                    f"pair {pair_id} is not fixed40: {len(r1_trials)}/{len(r2_trials)}"
                )
            r1_cache = subset_cache(raw[r1_session], r1_trials)
            r2_cache = subset_cache(raw[r2_session], r2_trials)
            for direction, train_cache, test_cache in (
                ("R1->R2", r1_cache, r2_cache),
                ("R2->R1", r2_cache, r1_cache),
            ):
                actual, predicted, meta = decode_with_predictions(
                    train_cache, test_cache
                )
                profile = time_resolved_metrics(actual, predicted, meta)
                profile.insert(
                    0,
                    "test_session",
                    r2_session if direction == "R1->R2" else r1_session,
                )
                profile.insert(
                    0,
                    "train_session",
                    r1_session if direction == "R1->R2" else r2_session,
                )
                profile.insert(0, "direction", direction)
                profile.insert(0, "r2_session", r2_session)
                profile.insert(0, "r1_session", r1_session)
                profile.insert(0, "pair_id", pair_id)
                profile.insert(0, "animal", "TS")
                rows.append(profile)
            print(f"[TS joint fixed40] pair {pair_id:02d}/42 complete", flush=True)

    profiles = pd.concat(rows, ignore_index=True)
    windows = summarize_windows(profiles)
    gaps = summarize_directional_gaps(profiles)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(OUT_PROFILES, index=False)
    windows.to_csv(OUT_WINDOWS, index=False)
    gaps.to_csv(OUT_GAPS, index=False)
    plot_profiles(profiles, OUT_FIGURE, "TS joint fixed40")
    print("\nDirectional-gap summary")
    print(gaps.round(6).to_string(index=False))
    print(f"\nsaved {OUT_PROFILES}")
    print(f"saved {OUT_WINDOWS}")
    print(f"saved {OUT_GAPS}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
