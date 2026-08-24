"""Validate, summarize, and plot the fixed-common-target source test."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="marmoset_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="marmoset_xdg_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, str(THIS_DIR.parents[1]))
sys.path.insert(0, str(THIS_DIR))

from big_sweep_phase2_crossday import ANIMAL_SESSIONS
from common_target_source_test import CONDITIONS
from decoder_consensus_crossanimal import PRIMARY_CONDITIONS


REPO = THIS_DIR.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry" / "common_target_source_test"
FIGURE = (
    REPO / "Results" / "manifold_geometry" / "figures"
    / "fig_common_target_source_test.png"
)
TARGETS = ("relative_position", "relative_velocity")
N_REPEATS = 5
N_FOLDS = 5
N_CONDITIONS = len(CONDITIONS)


def expected_paths() -> list[Path]:
    paths = []
    for target in TARGETS:
        for target_epoch in ("R1", "R2"):
            for job in range(2):
                paths.append(
                    OUT_DIR
                    / f"common_target_ts_{target}_{target_epoch.lower()}_"
                    f"job_{job}_of_2.csv"
                )
        paths.append(
            OUT_DIR
            / f"common_target_ty_{target}_r1_job_0_of_1.csv"
        )
    return paths


def load_and_validate() -> pd.DataFrame:
    paths = expected_paths()
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing common-target shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    ts_r1, ts_r2 = ANIMAL_SESSIONS["TS"]
    ty_r1, ty_r2 = ANIMAL_SESSIONS["TY"]
    triads_per_target = (
        2 * len(ts_r1) * len(ts_r2) + len(ty_r1) * len(ty_r2)
    )
    expected = (
        len(TARGETS) * triads_per_target * N_REPEATS * N_FOLDS
        * N_CONDITIONS
    )
    if len(frame) != expected:
        raise AssertionError(
            f"expected {expected} common-target rows, found {len(frame)}"
        )
    keys = [
        "animal", "target", "target_epoch", "r1_source", "r2_source",
        "common_target", "repeat", "fold", "decoder", "variant",
    ]
    if frame.duplicated(keys).any():
        raise AssertionError("duplicate common-target decoder rows")
    if (frame.common_target == frame.r1_source).any():
        raise AssertionError("R1 source reused as common target")
    if (frame.common_target == frame.r2_source).any():
        raise AssertionError("R2 source reused as common target")
    return frame


def summarize_conditions(frame: pd.DataFrame):
    keys = [
        "animal", "target", "target_epoch", "decoder", "variant",
        "r1_source", "r2_source", "common_target",
    ]
    triad = frame.groupby(keys, as_index=False)[
        ["r1_source_score", "r2_source_score", "source_advantage"]
    ].mean()
    by_target = triad.groupby(
        ["animal", "target", "target_epoch", "decoder", "variant",
         "common_target"],
        as_index=False,
    )[["r1_source_score", "r2_source_score", "source_advantage"]].mean()
    summary = triad.groupby(
        ["animal", "target", "target_epoch", "decoder", "variant"],
        as_index=False,
    ).agg(
        r1_source_score=("r1_source_score", "mean"),
        r2_source_score=("r2_source_score", "mean"),
        source_advantage=("source_advantage", "mean"),
        triad_advantage_sd=("source_advantage", "std"),
        positive_triad_fraction=("source_advantage", lambda x: float((x > 0).mean())),
        n_triads=("source_advantage", "size"),
        n_common_targets=("common_target", "nunique"),
    )
    return triad, by_target, summary


def summarize_consensus(frame: pd.DataFrame):
    included = frame.consensus_included
    if included.dtype != bool:
        included = included.astype(str).str.lower().map({"true": True, "false": False})
    primary = frame[included].copy()
    split_keys = [
        "animal", "target", "target_epoch", "r1_source", "r2_source",
        "common_target", "repeat", "fold",
    ]
    split = primary.groupby(split_keys).agg(
        mean_source_advantage=("source_advantage", "mean"),
        median_source_advantage=("source_advantage", "median"),
        n_positive_decoders=("source_advantage", lambda x: int((x > 0).sum())),
        decoder_sd=("source_advantage", lambda x: float(np.std(x))),
        n_decoders=("source_advantage", "size"),
    ).reset_index()
    if set(split.n_decoders) != {len(PRIMARY_CONDITIONS)}:
        raise AssertionError("common-target consensus does not contain four decoders")
    triad = split.groupby(
        ["animal", "target", "target_epoch", "r1_source", "r2_source",
         "common_target"],
        as_index=False,
    )[
        ["mean_source_advantage", "median_source_advantage",
         "n_positive_decoders", "decoder_sd"]
    ].mean()
    by_target = triad.groupby(
        ["animal", "target", "target_epoch", "common_target"],
        as_index=False,
    )[
        ["mean_source_advantage", "median_source_advantage",
         "n_positive_decoders", "decoder_sd"]
    ].mean()
    summary = triad.groupby(
        ["animal", "target", "target_epoch"], as_index=False
    ).agg(
        mean_source_advantage=("mean_source_advantage", "mean"),
        median_source_advantage=("median_source_advantage", "mean"),
        n_positive_decoders=("n_positive_decoders", "mean"),
        decoder_sd=("decoder_sd", "mean"),
        positive_triad_fraction=(
            "median_source_advantage", lambda x: float((x > 0).mean())
        ),
        n_triads=("median_source_advantage", "size"),
        n_common_targets=("common_target", "nunique"),
    )
    return split, triad, by_target, summary


def plot(condition_triad: pd.DataFrame, condition_summary: pd.DataFrame):
    labels = {
        ("ridge", "instantaneous"): "Ridge",
        ("wiener", "history_2"): "Wiener",
        ("arx", "trial_aware"): "ARX\ntrial",
        ("kalman", "original_concatenated"): "Kalman\nconcat",
        ("kalman", "trial_aware"): "Kalman\ntrial",
        ("kalman", "behaviour_center"): "Kalman\ncenter",
    }
    colors = ("#0072B2", "#56B4E9", "#E69F00", "#CC79A7", "#D55E00", "#009E73")
    panels = (("TS", "R1"), ("TS", "R2"), ("TY", "R1"))
    rng = np.random.default_rng(20260808)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), sharey=True)
    for ax, (animal, target_epoch) in zip(axes, panels, strict=True):
        triad = condition_triad[
            (condition_triad.animal == animal)
            & (condition_triad.target == "relative_position")
            & (condition_triad.target_epoch == target_epoch)
        ]
        summary = condition_summary[
            (condition_summary.animal == animal)
            & (condition_summary.target == "relative_position")
            & (condition_summary.target_epoch == target_epoch)
        ].set_index(["decoder", "variant"])
        x = np.arange(len(CONDITIONS))
        means = [summary.at[condition, "source_advantage"] for condition in CONDITIONS]
        ax.bar(x, means, color=colors, alpha=0.75, edgecolor="black", linewidth=0.5)
        for index, condition in enumerate(CONDITIONS):
            values = triad[
                (triad.decoder == condition[0]) & (triad.variant == condition[1])
            ].source_advantage.to_numpy()
            ax.scatter(
                index + rng.uniform(-0.1, 0.1, len(values)), values,
                color=colors[index], alpha=0.22, s=12, linewidths=0,
            )
        ax.axhline(0, color="0.25", linewidth=1)
        ax.set_xticks(x, [labels[c] for c in CONDITIONS])
        ax.set_title(
            f"{animal}: common {target_epoch} target",
            loc="left", weight="bold",
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("R2-source − R1-source correlation\non identical target trials")
    fig.suptitle(
        "Fixed-common-target test isolates the source side of transfer",
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    frame = load_and_validate()
    condition_triad, condition_target, condition_summary = summarize_conditions(frame)
    consensus_split, consensus_triad, consensus_target, consensus_summary = (
        summarize_consensus(frame)
    )
    outputs = {
        "common_target_all.csv": frame,
        "common_target_condition_triads.csv": condition_triad,
        "common_target_condition_by_target_session.csv": condition_target,
        "common_target_condition_summary.csv": condition_summary,
        "common_target_consensus_splits.csv": consensus_split,
        "common_target_consensus_triads.csv": consensus_triad,
        "common_target_consensus_by_target_session.csv": consensus_target,
        "common_target_consensus_summary.csv": consensus_summary,
    }
    for filename, output in outputs.items():
        output.to_csv(OUT_DIR / filename, index=False)
    plot(condition_triad, condition_summary)
    print("CONDITIONS")
    print(condition_summary.round(4).to_string(index=False))
    print("\nFOUR-DECODER CONSENSUS")
    print(consensus_summary.round(4).to_string(index=False))
    print(f"\nsaved {FIGURE}")


if __name__ == "__main__":
    main()
