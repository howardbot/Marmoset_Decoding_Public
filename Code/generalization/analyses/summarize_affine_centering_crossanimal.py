"""Summarize the cross-fitted affine-centering falsifier in TS and TY."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="marmoset_mpl_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="marmoset_xdg_"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parents[2]
IN_DIR = REPO / "Results" / "manifold_geometry" / "h_observation_fine_swap"
JOB_COUNTS = {"TS": 8, "TY": 6}
CONDITIONS = (
    "original_concatenated", "original_trial_aware", "source_b",
    "behaviour_center", "target_b",
)
PLOT_CONDITIONS = (
    "original_concatenated", "original_trial_aware",
    "behaviour_center", "target_b",
)
OUT_SPLITS = IN_DIR / "affine_centering_crossanimal_splits.csv"
OUT_PAIRS = IN_DIR / "affine_centering_crossanimal_pair_means.csv"
OUT_SUMMARY = IN_DIR / "affine_centering_crossanimal_summary.csv"
OUT_FIGURE = (
    REPO / "Results" / "manifold_geometry" / "figures"
    / "fig_affine_centering_crossanimal.png"
)


def job_paths(animal: str) -> list[Path]:
    count = JOB_COUNTS[animal]
    animal_tag = "" if animal == "TS" else f"_{animal.lower()}"
    return [
        IN_DIR / f"h_centering{animal_tag}_job_{index}_of_{count}.csv"
        for index in range(count)
    ]


def load_animal(animal: str) -> pd.DataFrame:
    paths = job_paths(animal)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {animal} centering jobs: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if "animal" not in frame:
        frame["animal"] = animal
    elif set(frame.animal) != {animal}:
        raise AssertionError(f"unexpected animal labels in {animal} jobs")
    return frame


def prepare_splits(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "animal", "q_context", "condition", "r1_session", "r2_session",
        "repeat", "fold",
    ]
    required = set(keys + ["direction", "score"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"centering rows missing columns: {sorted(missing)}")
    if frame[keys + ["direction"]].duplicated().any():
        raise AssertionError("duplicate affine-centering direction scores")
    wide = frame.pivot(index=keys, columns="direction", values="score").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"forward": "forward_corr", "reverse": "reverse_corr"})
    if wide[["forward_corr", "reverse_corr"]].isna().any().any():
        raise AssertionError("direction pivot produced missing scores")
    wide["directional_gap"] = wide.reverse_corr - wide.forward_corr
    baseline = wide[wide.condition == "source_b"][[
        "animal", "q_context", "r1_session", "r2_session", "repeat", "fold",
        "forward_corr", "reverse_corr", "directional_gap",
    ]].rename(columns={
        "forward_corr": "baseline_forward_corr",
        "reverse_corr": "baseline_reverse_corr",
        "directional_gap": "baseline_gap",
    })
    baseline_keys = [
        "animal", "q_context", "r1_session", "r2_session", "repeat", "fold"
    ]
    if baseline[baseline_keys].duplicated().any():
        raise AssertionError("source-b baseline is not unique")
    wide = wide.merge(baseline, on=baseline_keys, validate="many_to_one")
    wide["forward_change"] = wide.forward_corr - wide.baseline_forward_corr
    wide["reverse_change"] = wide.reverse_corr - wide.baseline_reverse_corr
    wide["gap_closed"] = wide.baseline_gap - wide.directional_gap
    original = wide[wide.condition == "original_concatenated"][[
        "animal", "r1_session", "r2_session", "repeat", "fold",
        "forward_corr", "reverse_corr", "directional_gap",
    ]].rename(columns={
        "forward_corr": "original_forward_corr",
        "reverse_corr": "original_reverse_corr",
        "directional_gap": "original_gap",
    })
    original_keys = [
        "animal", "r1_session", "r2_session", "repeat", "fold"
    ]
    if original[original_keys].duplicated().any():
        raise AssertionError("original concatenated baseline is not unique")
    wide = wide.merge(original, on=original_keys, validate="many_to_one")
    wide["gap_closed_from_original"] = wide.original_gap - wide.directional_gap
    return wide


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "forward_corr", "reverse_corr", "directional_gap",
        "baseline_forward_corr", "baseline_reverse_corr", "baseline_gap",
        "forward_change", "reverse_change", "gap_closed",
        "original_forward_corr", "original_reverse_corr", "original_gap",
        "gap_closed_from_original",
    ]
    pair_keys = ["animal", "q_context", "condition", "r1_session", "r2_session"]
    pairs = frame.groupby(pair_keys, as_index=False)[metrics].mean()
    summary = pairs.groupby(["animal", "q_context", "condition"], as_index=False).agg(
        forward_corr=("forward_corr", "mean"),
        reverse_corr=("reverse_corr", "mean"),
        directional_gap=("directional_gap", "mean"),
        baseline_gap=("baseline_gap", "mean"),
        forward_change=("forward_change", "mean"),
        reverse_change=("reverse_change", "mean"),
        gap_closed=("gap_closed", "mean"),
        original_gap=("original_gap", "mean"),
        gap_closed_from_original=("gap_closed_from_original", "mean"),
        pair_gap_sd=("directional_gap", "std"),
        n_pairs=("directional_gap", "size"),
    )
    summary["gap_fraction_closed"] = np.where(
        np.abs(summary.baseline_gap) > 1e-12,
        summary.gap_closed / summary.baseline_gap,
        np.nan,
    )
    summary["gap_fraction_closed_from_original"] = np.where(
        np.abs(summary.original_gap) > 1e-12,
        summary.gap_closed_from_original / summary.original_gap,
        np.nan,
    )
    return pairs, summary


def plot(pairs: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    source_q = summary[summary.q_context == "source"].copy()
    pair_source_q = pairs[pairs.q_context == "source"]
    labels = {
        "original_concatenated": "Original\nconcatenated",
        "original_trial_aware": "Trial-aware\nno intercept",
        "source_b": "Source\nintercept",
        "behaviour_center": "Behavior\ncenter",
        "target_b": "Target\nintercept",
    }
    colors = {
        "original_concatenated": "#0072B2",
        "original_trial_aware": "#E69F00",
        "source_b": "#666666",
        "behaviour_center": "#009E73",
        "target_b": "#CC79A7",
    }
    rng = np.random.default_rng(20260808)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    for ax, animal in zip(axes, ("TS", "TY"), strict=True):
        animal_summary = source_q[source_q.animal == animal].set_index("condition")
        animal_pairs = pair_source_q[pair_source_q.animal == animal]
        means = [
            float(animal_summary.at[condition, "directional_gap"])
            for condition in PLOT_CONDITIONS
        ]
        ax.bar(
            np.arange(len(PLOT_CONDITIONS)), means,
            color=[colors[condition] for condition in PLOT_CONDITIONS],
            width=0.64, edgecolor="black", linewidth=0.6, alpha=0.78,
        )
        for index, condition in enumerate(PLOT_CONDITIONS):
            values = animal_pairs[animal_pairs.condition == condition].directional_gap.to_numpy()
            ax.scatter(
                index + rng.uniform(-0.11, 0.11, len(values)), values,
                color=colors[condition], s=16, alpha=0.35, linewidths=0,
            )
            value = means[index]
            ax.text(
                index, value + (0.008 if value >= 0 else -0.008), f"{value:+.3f}",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=9,
            )
        ax.axhline(0.0, color="0.25", linewidth=1.0)
        ax.set_xticks(
            np.arange(len(PLOT_CONDITIONS)),
            [labels[c] for c in PLOT_CONDITIONS],
        )
        ax.set_title(f"{animal}: position", loc="left", weight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylabel("R2→R1 − R1→R2 correlation")
    fig.suptitle(
        "Cross-fitted transition/reference-frame falsifier\n"
        "Concatenated → trial-aware transitions → affine behavior center",
        weight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Dots are R1/R2 session pairs (TS: 14×3; TY: 6×1). "
        "TY is descriptive because n(R2)=1.",
        ha="center", fontsize=8.5, color="0.35",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    raw = pd.concat([load_animal(animal) for animal in JOB_COUNTS], ignore_index=True)
    splits = prepare_splits(raw)
    pairs, summary = summarize(splits)
    OUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(OUT_SPLITS, index=False)
    pairs.to_csv(OUT_PAIRS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    plot(pairs, summary, OUT_FIGURE)
    print(summary.round(4).to_string(index=False))
    print(f"saved {OUT_SPLITS}")
    print(f"saved {OUT_PAIRS}")
    print(f"saved {OUT_SUMMARY}")
    print(f"saved {OUT_FIGURE}")


if __name__ == "__main__":
    main()
