"""Canonical project paths, session grids, and locked decoder settings.

This module is the single source of truth for the active cross-day analyses.
Analysis scripts may expose backward-compatible aliases, but they should not
maintain independent copies of session identifiers or repository paths.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "Code" / "generalization"
DATA_DIR = REPO_ROOT / "Data"
RESULTS_DIR = REPO_ROOT / "Results"
CURRENT_RESULTS_DIR = RESULTS_DIR / "current"
WORKFLOW_RESULTS_DIR = RESULTS_DIR / "workflows"
ARCHIVE_RESULTS_DIR = RESULTS_DIR / "archive"
GENERALIZATION_RESULTS_DIR = WORKFLOW_RESULTS_DIR / "generalization"
MANIFOLD_RESULTS_DIR = WORKFLOW_RESULTS_DIR / "manifold_geometry"
FORGET_CONTROL_RESULTS_DIR = (
    CURRENT_RESULTS_DIR / "forget_control" / "equal_n_3x3" / "tables"
)
REPORTS_DIR = REPO_ROOT / "Reports"


# TS original interference experiment: 14 R1 static-task dates and 3 R2 dates.
TS_INTERFERENCE_R1 = (
    "TSAL20250731_0830_staticAndStaticFree",
    "TSAL20250801_0830_staticAndStaticFree",
    "TSAL20250802_0830_staticAndStaticFree001",
    "TSAL20250803_1400_staticAndStaticFree001",
    "TSAL20250804_0830_staticAndStaticFree001",
    "TSAL20250805_0830_staticAndStaticFree001",
    "TSAL20250806_0830_staticAndStaticFree001",
    "TSAL20250807_0830_staticAndStaticFree001",
    "TSAL20250808_0830_staticAndStaticFree001",
    "TSAL20250809_0830_staticAndStaticFree001",
    "TSAL20250810_0830_staticAndStaticFree001",
    "TSAL20250811_0830_staticAndStaticFree001",
    "TSAL20250812_0830_staticAndStaticFree001",
    "TSAL20250813_0830_staticAndStaticFree001",
)
TS_INTERFERENCE_R2 = (
    "TSAL20250828_0830_interferenceAndInterferenceFree001",
    "TSAL20250829_0830_interferenceAndInterferenceFree001",
    "TSAL20250830_0830_interferenceAndInterferenceFree001",
)


# TY replication experiment: 11 available R1 dates and all 3 R2 dates.
TY_INTERFERENCE_R1 = (
    "TYTR20250206_0830_staticAndStaticFree001",
    "TYTR20250207_0830_staticAndStaticFree001",
    "TYTR20250208_0830_staticAndStaticFree001",
    "TYTR20250209_0830_staticAndStaticFree",
    "TYTR20250210_0830_staticAndStaticFree001",
    "TYTR20250212_0830_staticAndStaticFree001",
    "TYTR20250213_0830_staticAndStaticFree001",
    "TYTR20250215_0830_staticAndStaticFree001",
    "TYTR20250216_0830_staticAndStaticFree001",
    "TYTR20250217_0830_staticAndStaticFree001",
    "TYTR20250218_0830_staticAndStaticFree001",
)
TY_INTERFERENCE_R2 = (
    "TYTR20250304_0830_interferenceAndInterferenceFree001",
    "TYTR20250305_0830_interferenceAndInterferenceFree001",
    "TYTR20250306_0830_interferenceAndInterferenceFree001",
)


# TS no-interference forget control: complete current 3 x 3 processed grid.
TS_FORGET_R1 = (
    "TSAL20260609_0830_forgetAndForgetFree001",
    "TSAL20260610_0830_forgetAndForgetFree001",
    "TSAL20260611_0830_forgetAndForgetFree001",
)
TS_FORGET_R2 = (
    "TSAL20260626_0830_forgetAndForgetFree001",
    "TSAL20260627_0830_forgetAndForgetFree001",
    "TSAL20260628_0830_forgetAndForgetFree001",
)


INTERFERENCE_SESSIONS = {
    "TS": (TS_INTERFERENCE_R1, TS_INTERFERENCE_R2),
    "TY": (TY_INTERFERENCE_R1, TY_INTERFERENCE_R2),
}
FORGET_CONTROL_SESSIONS = {"TS": (TS_FORGET_R1, TS_FORGET_R2)}


# Diagnosed pose-tracking outlier in the original TS interference experiment.
EXCLUDE_TRIALS = {
    "TSAL20250828_0830_interferenceAndInterferenceFree001": (41,),
}


LOCKED_DECODER = {
    "bin_size_ms": 30,
    "smoother": "butter_o2",
    "lag_ms": 0,
    "n_pcs": 12,
    "n_phase_bins": 30,
    "decoder": "kalman",
    "target_mode": "relative_position",
    "trial_window": "start_to_peak",
}


def nwb_path(session: str) -> Path:
    """Return the canonical processed-NWB path for a session identifier."""
    return DATA_DIR / f"{session}_processed.nwb"


def session_date(session: str) -> str:
    """Extract YYYYMMDD from a TSAL/TYTR session identifier."""
    prefix_length = 4
    return session[prefix_length : prefix_length + 8]
