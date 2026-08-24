"""Single source of truth for figure configuration and shared plotting helpers.

All plot scripts in this directory must import LOCKED_CONFIG from here and
filter the long sweep CSV through ``filter_locked()`` before plotting. Any
figure that deviates from LOCKED_CONFIG must do so via an explicit override
and label the deviation in its caption.

LOCKED_CONFIG was selected after Phase 1 (within-day) and Phase 2 (cross-day)
parameter sweeps. Rationale:

  bin_size_ms=30
      Near-best Kalman cross-day for both position and velocity;
      matches Gallego 2020 standard; matches the project's cross_day_decoder
      locked config.
  smoother="butter_o2"
      Marginally best on velocity (+5% over savgol); butter_o4 is
      indistinguishable from butter_o2; matches Winter biomechanics
      "4th-order zero-lag" convention (order=2 + filtfilt).
  decoder="kalman"
      Best decoder overall; state-space prior acts as soft regularization.
  lag_ms=0
      Strongest cross-day signal for position; velocity is essentially flat
      around 0; cleanest interpretation (no neural-motor lead assumption).
  outlier_mode="exclude"
      Required: 0828 trial 41 has a tracking failure that pulls
      0828-involving M2 down by ~50%. See compare_0813_vs_0828.py.
  target_mode="relative_velocity"
      Primary target tied to Aim 3 kinematic generalization story.
      relative_position is the SECONDARY_TARGET and gets a parallel panel.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from project_config import (
    GENERALIZATION_RESULTS_DIR,
    REPO_ROOT,
    TS_INTERFERENCE_R1,
    TS_INTERFERENCE_R2,
)


SWEEP_CSV = GENERALIZATION_RESULTS_DIR / "big_sweep_crossday_long.csv"
FIG_DIR = GENERALIZATION_RESULTS_DIR / "figures"


LOCKED_CONFIG = dict(
    bin_size_ms=30,
    smoother="butter_o2",
    target_mode="relative_velocity",
    decoder="kalman",
    lag_ms=0,
    outlier_mode="exclude",
)
SECONDARY_TARGET = "relative_position"
WIENER_HISTORY_MS = 50  # used only when decoder='wiener'

# Session order = R1 sorted chronologically, then R2.
SESSIONS_R1 = list(TS_INTERFERENCE_R1)
SESSIONS_R2 = list(TS_INTERFERENCE_R2)
ALL_SESSIONS = SESSIONS_R1 + SESSIONS_R2

# Index where R2 begins in ALL_SESSIONS. Used to draw the R1/R2 split line on
# the 17x17 cross-day matrix.
R2_BOUNDARY_INDEX = len(SESSIONS_R1)  # = 14

# Short display labels (MMDD) for axis ticks.
SESSION_LABELS = [s.split("_")[0][-4:] for s in ALL_SESSIONS]  # e.g. "0731"


# --------------------------- Colors --------------------------- #
CMAP_PRIMARY = "viridis"      # sequential, perceptual; main heatmaps
CMAP_DIVERGING = "RdBu_r"     # diverging; delta / asymmetry maps

DIAGONAL_EDGE_COLOR = "red"
DIAGONAL_EDGE_WIDTH = 1.4
R1R2_SPLIT_COLOR = "white"
R1R2_SPLIT_WIDTH = 2.5

# Color-coding for the four pair categories (used in box / strip plots).
PAIR_COLORS = {
    "R1->R1": "#7f8c8d",
    "R2->R2": "#34495e",
    "R1->R2": "#e74c3c",  # interference direction (forward) — red
    "R2->R1": "#3498db",  # reverse direction — blue
}


# Axis/colorbar label. The metric is the per-trial Pearson correlation averaged
# across trials, so we call it exactly that: "corr". We never spell out "M2"
# (jargon) nor "accuracy" (misleading -- it is a correlation, not a hit rate).
METRIC_LABEL = "corr"


# --------------------------- Helpers --------------------------- #
def epoch_of(session: str) -> str:
    """Return 'R1' or 'R2' for a session tag."""
    return "R2" if session in SESSIONS_R2 else "R1"


def forward_reverse_pairs(mat):
    """For every (R1 day, R2 day), return (r1_day, r2_day, forward, reverse).

    forward = mat[train=R1 day, test=R2 day]   (R1 -> R2)
    reverse = mat[train=R2 day, test=R1 day]   (R2 -> R1)

    These share the same two sessions, so they form a natural pair for showing
    directional asymmetry. Pairs with a non-finite entry on either side are
    skipped.
    """
    out = []
    for a in SESSIONS_R1:
        for b in SESSIONS_R2:
            fwd = mat.at[a, b]
            rev = mat.at[b, a]
            if np.isfinite(fwd) and np.isfinite(rev):
                out.append((a, b, float(fwd), float(rev)))
    return out


def block_mean(mat, train_epoch, test_epoch, exclude_diagonal=True):
    """Mean of one R1/R2 block of the matrix (optionally excluding the diagonal)."""
    rows = SESSIONS_R1 if train_epoch == "R1" else SESSIONS_R2
    cols = SESSIONS_R1 if test_epoch == "R1" else SESSIONS_R2
    vals = []
    for r in rows:
        for c in cols:
            if exclude_diagonal and r == c:
                continue
            v = mat.at[r, c]
            if np.isfinite(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def pair_category(train_session: str, test_session: str) -> str:
    """Return one of 'R1->R1', 'R1->R2', 'R2->R1', 'R2->R2'."""
    return f"{epoch_of(train_session)}->{epoch_of(test_session)}"


def session_date(session: str) -> datetime:
    """Parse the YYYYMMDD date embedded in a session tag (e.g. TSAL20250731_...)."""
    digits = session.replace("TSAL", "")[:8]
    return datetime.strptime(digits, "%Y%m%d")


def day_gap(session_a: str, session_b: str) -> int:
    """Absolute calendar-day gap between two sessions."""
    return abs((session_date(session_a) - session_date(session_b)).days)


def load_sweep(csv_path: Path = SWEEP_CSV) -> pd.DataFrame:
    """Load the Phase-2 long-form sweep CSV with a couple of useful derived columns."""
    df = pd.read_csv(csv_path)
    df["is_diag"] = df["train_session"] == df["test_session"]
    df["pair_category"] = [
        pair_category(t, e) for t, e in zip(df["train_session"], df["test_session"])
    ]
    return df


def filter_locked(df: pd.DataFrame, **overrides) -> pd.DataFrame:
    """Subset ``df`` to rows matching LOCKED_CONFIG (with field-level overrides).

    Pass ``smoother="savgol"`` etc. to override individual fields. Pass
    ``history_ms=50`` only when decoder='wiener' (Kalman rows have NaN history).

    ``outlier_mode`` semantics deserve a note. The sweep only stores the
    'exclude' variant for pair cells that involve 0828; non-0828 cells exist
    only with outlier_mode='include'. So when the caller asks for
    outlier_mode='exclude', we need to grab the 'exclude' row when it exists
    and fall back to 'include' otherwise -- otherwise we lose 95% of the
    matrix. We implement this by NOT filtering on outlier_mode directly when
    'exclude' is requested; instead we prefer 'exclude' rows per pair and use
    'include' as a fallback for pairs without an 'exclude' row.
    """
    cfg = {**LOCKED_CONFIG, **overrides}
    mask = pd.Series(True, index=df.index)
    for key, val in cfg.items():
        if key == "outlier_mode" or key not in df.columns:
            continue
        mask &= df[key] == val
    sub = df[mask].copy()
    # Wiener-only: also filter history_ms if asked.
    if cfg.get("decoder") == "wiener":
        hist_ms = cfg.get("history_ms", WIENER_HISTORY_MS)
        sub = sub[sub["history_ms"] == hist_ms].copy()

    requested = cfg.get("outlier_mode", "exclude")
    if requested == "include":
        # Caller explicitly wants the no-cleaning version.
        return sub[sub["outlier_mode"] == "include"].copy()
    # 'exclude': prefer 'exclude' row per (train, test, lag); fall back to 'include'.
    # The sweep guarantees at most one 'exclude' and one 'include' row per
    # (cell, train, test, lag, decoder, history) tuple, so picking with
    # outlier_mode rank 0=exclude / 1=include and keeping the first per group
    # works cleanly.
    sub["_pref"] = (sub["outlier_mode"] == "include").astype(int)  # 0 for exclude, 1 for include
    sub = sub.sort_values(["train_session", "test_session", "lag_ms",
                          "decoder", "history_ms", "_pref"])
    sub = sub.drop_duplicates(
        subset=["train_session", "test_session", "lag_ms", "decoder", "history_ms"],
        keep="first",
    ).drop(columns="_pref")
    return sub


def pivot_matrix(df_filtered: pd.DataFrame, value_col: str = "M2_mean") -> pd.DataFrame:
    """Return a 15x15 matrix (train rows, test cols) in ALL_SESSIONS order.

    Cells with no matching row become NaN. Diagonal entries come from same-session
    rows already in the sweep CSV.
    """
    mat = df_filtered.pivot(
        index="train_session", columns="test_session", values=value_col,
    )
    return mat.reindex(index=ALL_SESSIONS, columns=ALL_SESSIONS)


def draw_diagonal_frames(ax, n: int = len(ALL_SESSIONS)) -> None:
    """Outline diagonal cells (within-day) in red on a heatmap axes."""
    import matplotlib.patches as patches
    for i in range(n):
        ax.add_patch(patches.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False,
            edgecolor=DIAGONAL_EDGE_COLOR,
            linewidth=DIAGONAL_EDGE_WIDTH,
        ))


def draw_r1r2_split(ax, n: int = len(ALL_SESSIONS), boundary: int = R2_BOUNDARY_INDEX) -> None:
    """Draw thick split lines between R1 and R2 blocks (both axes)."""
    # Vertical line between columns boundary-1 and boundary.
    ax.axvline(boundary - 0.5, color=R1R2_SPLIT_COLOR, linewidth=R1R2_SPLIT_WIDTH)
    # Horizontal line between rows boundary-1 and boundary.
    ax.axhline(boundary - 0.5, color=R1R2_SPLIT_COLOR, linewidth=R1R2_SPLIT_WIDTH)


def set_session_ticks(ax) -> None:
    """Standard tick formatting for the 15x15 matrix."""
    n = len(ALL_SESSIONS)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SESSION_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(SESSION_LABELS, fontsize=8)


def shared_vmax(*matrices, quantile: float = 0.99) -> float:
    """Pick a shared vmax across multiple matrices using a high quantile so a
    single outlier cell does not crush the rest of the colormap."""
    stacked = np.concatenate([np.asarray(m).ravel() for m in matrices])
    stacked = stacked[np.isfinite(stacked)]
    if stacked.size == 0:
        return 1.0
    return float(np.nanquantile(stacked, quantile))


def ensure_fig_dir() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def config_caption(**overrides) -> str:
    """One-line config string for figure captions. Always shown."""
    cfg = {**LOCKED_CONFIG, **overrides}
    return (
        f"bin={cfg['bin_size_ms']}ms, smoother={cfg['smoother']}, "
        f"decoder={cfg['decoder']}, lag={cfg['lag_ms']}ms, "
        f"target={cfg['target_mode']}, outlier_mode={cfg['outlier_mode']}"
    )
