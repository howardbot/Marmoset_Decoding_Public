"""Validate and summarize cross-animal decoder consensus and map stability."""
from __future__ import annotations

import argparse
import itertools
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
from decoder_consensus_crossanimal import PRIMARY_CONDITIONS


REPO = Path(__file__).resolve().parent.parents[2]
CONSENSUS_DIR = (
    REPO / "Results" / "manifold_geometry" / "decoder_consensus_crossanimal"
)
MAPPING_DIR = REPO / "Results" / "manifold_geometry" / "shared_mapping_stability"
FIGURE = (
    REPO / "Results" / "manifold_geometry" / "figures"
    / "fig_decoder_consensus_crossanimal.png"
)
N_REPEATS = 5
N_FOLDS = 5
N_CONDITIONS = 6
CONDITION_ORDER = (
    ("ridge", "instantaneous"),
    ("wiener", "history_2"),
    ("arx", "trial_aware"),
    ("kalman", "original_concatenated"),
    ("kalman", "trial_aware"),
    ("kalman", "behaviour_center"),
)
MAPPING_METRICS = (
    "within_corr", "cv_r2", "weight_cosine", "prediction_agreement",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets", nargs="+", default=["relative_position"],
        choices=("relative_position", "relative_velocity"),
    )
    return parser.parse_args()


def load_consensus(targets: tuple[str, ...]) -> pd.DataFrame:
    paths = []
    expected = 0
    for animal, (r1_sessions, r2_sessions) in ANIMAL_SESSIONS.items():
        for target in targets:
            for r2_index in range(len(r2_sessions)):
                paths.append(
                    CONSENSUS_DIR
                    / f"consensus_{animal.lower()}_{target}_r2_{r2_index}.csv"
                )
            expected += (
                len(r1_sessions) * len(r2_sessions) * N_REPEATS * N_FOLDS
                * N_CONDITIONS
            )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing decoder-consensus shards: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(frame) != expected:
        raise AssertionError(
            f"expected {expected} decoder-consensus rows, found {len(frame)}"
        )
    keys = [
        "animal", "target", "r1_session", "r2_session", "repeat", "fold",
        "decoder", "variant",
    ]
    if frame.duplicated(keys).any():
        raise AssertionError("duplicate decoder-consensus rows")
    return frame


def add_transfer_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for support in ("native", "common"):
        frame[f"gap_{support}"] = (
            frame[f"rev_cross_{support}"] - frame[f"fwd_cross_{support}"]
        )
        frame[f"own_advantage_{support}"] = (
            frame[f"rev_own_{support}"] - frame[f"fwd_own_{support}"]
        )
        frame[f"target_difficulty_gap_{support}"] = (
            frame[f"fwd_own_{support}"] - frame[f"rev_own_{support}"]
        )
        frame[f"excess_gap_{support}"] = (
            frame[f"gap_{support}"] - frame[f"target_difficulty_gap_{support}"]
        )
        valid_forward = frame[f"fwd_own_{support}"].abs() > 0.05
        valid_reverse = frame[f"rev_own_{support}"].abs() > 0.05
        frame[f"fwd_retention_{support}"] = np.where(
            valid_forward,
            frame[f"fwd_cross_{support}"] / frame[f"fwd_own_{support}"],
            np.nan,
        )
        frame[f"rev_retention_{support}"] = np.where(
            valid_reverse,
            frame[f"rev_cross_{support}"] / frame[f"rev_own_{support}"],
            np.nan,
        )
        frame[f"retention_gap_{support}"] = (
            frame[f"rev_retention_{support}"]
            - frame[f"fwd_retention_{support}"]
        )
        frame[f"fwd_target_retention_{support}"] = np.where(
            valid_reverse,
            frame[f"fwd_cross_{support}"] / frame[f"rev_own_{support}"],
            np.nan,
        )
        frame[f"rev_target_retention_{support}"] = np.where(
            valid_forward,
            frame[f"rev_cross_{support}"] / frame[f"fwd_own_{support}"],
            np.nan,
        )
        frame[f"target_retention_gap_{support}"] = (
            frame[f"rev_target_retention_{support}"]
            - frame[f"fwd_target_retention_{support}"]
        )
    return frame


def condition_summaries(frame: pd.DataFrame):
    metrics = [
        "fwd_cross_common", "rev_cross_common", "gap_common",
        "fwd_own_common", "rev_own_common", "own_advantage_common",
        "fwd_retention_common", "rev_retention_common", "retention_gap_common",
        "target_difficulty_gap_common", "excess_gap_common",
        "fwd_target_retention_common", "rev_target_retention_common",
        "target_retention_gap_common",
    ]
    pair_keys = [
        "animal", "target", "decoder", "variant", "r1_session", "r2_session"
    ]
    pairs = frame.groupby(pair_keys, as_index=False)[metrics].mean()
    day = pairs.groupby(
        ["animal", "target", "decoder", "variant", "r2_session"],
        as_index=False,
    )[metrics].mean()
    summary = pairs.groupby(
        ["animal", "target", "decoder", "variant"], as_index=False
    ).agg(
        **{metric: (metric, "mean") for metric in metrics},
        pair_gap_sd=("gap_common", "std"),
        n_pairs=("gap_common", "size"),
    )
    return pairs, day, summary


def decoder_consensus(frame: pd.DataFrame):
    included = frame.consensus_included
    if included.dtype != bool:
        included = included.astype(str).str.lower().map({"true": True, "false": False})
    if included.isna().any():
        raise ValueError("invalid consensus_included labels")
    primary = frame[included].copy()
    keys = [
        "animal", "target", "r1_session", "r2_session", "repeat", "fold"
    ]
    grouped = primary.groupby(keys)
    split = grouped.agg(
        consensus_gap=("gap_common", "mean"),
        consensus_median_gap=("gap_common", "median"),
        consensus_excess_gap=("excess_gap_common", "mean"),
        consensus_median_excess_gap=("excess_gap_common", "median"),
        consensus_target_retention_gap=("target_retention_gap_common", "mean"),
        consensus_retention_gap=("retention_gap_common", "mean"),
        decoder_gap_sd=("gap_common", lambda values: float(np.std(values))),
        n_positive_decoders=("gap_common", lambda values: int((values > 0).sum())),
        all_decoders_positive=("gap_common", lambda values: bool((values > 0).all())),
        all_decoders_negative=("gap_common", lambda values: bool((values < 0).all())),
        n_positive_excess_decoders=(
            "excess_gap_common", lambda values: int((values > 0).sum())
        ),
        mean_source_quality_advantage=("own_advantage_common", "mean"),
        n_decoders=("gap_common", "size"),
    ).reset_index()
    if set(split.n_decoders) != {len(PRIMARY_CONDITIONS)}:
        raise AssertionError("decoder consensus does not contain four families")
    pair = split.groupby(
        ["animal", "target", "r1_session", "r2_session"], as_index=False
    )[
        ["consensus_gap", "consensus_median_gap", "consensus_excess_gap",
         "consensus_median_excess_gap", "consensus_target_retention_gap",
         "consensus_retention_gap", "decoder_gap_sd", "n_positive_decoders",
         "n_positive_excess_decoders", "all_decoders_positive",
         "all_decoders_negative", "mean_source_quality_advantage"]
    ].mean()
    day = pair.groupby(
        ["animal", "target", "r2_session"], as_index=False
    )[
        ["consensus_gap", "consensus_median_gap", "consensus_excess_gap",
         "consensus_median_excess_gap", "consensus_target_retention_gap",
         "consensus_retention_gap", "decoder_gap_sd", "n_positive_decoders",
         "n_positive_excess_decoders", "all_decoders_positive",
         "all_decoders_negative", "mean_source_quality_advantage"]
    ].mean()
    summary = pair.groupby(["animal", "target"], as_index=False).agg(
        consensus_gap=("consensus_gap", "mean"),
        consensus_median_gap=("consensus_median_gap", "mean"),
        consensus_excess_gap=("consensus_excess_gap", "mean"),
        consensus_median_excess_gap=("consensus_median_excess_gap", "mean"),
        consensus_target_retention_gap=("consensus_target_retention_gap", "mean"),
        consensus_retention_gap=("consensus_retention_gap", "mean"),
        decoder_gap_sd=("decoder_gap_sd", "mean"),
        n_positive_decoders=("n_positive_decoders", "mean"),
        n_positive_excess_decoders=("n_positive_excess_decoders", "mean"),
        all_decoders_positive_fraction=("all_decoders_positive", "mean"),
        all_decoders_negative_fraction=("all_decoders_negative", "mean"),
        mean_source_quality_advantage=("mean_source_quality_advantage", "mean"),
        n_pairs=("consensus_gap", "size"),
    )
    return split, pair, day, summary


def exact_label_permutation(
    values: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Return observed R2-R1 difference and exhaustive two-sided label p."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    n_r2 = int(np.sum(labels == "R2"))
    if n_r2 == 0 or n_r2 == len(values):
        raise ValueError("both R1 and R2 labels are required")
    observed = float(values[labels == "R2"].mean() - values[labels == "R1"].mean())
    differences = []
    indices = np.arange(len(values))
    for r2_indices in itertools.combinations(indices, n_r2):
        r2_mask = np.zeros(len(values), dtype=bool)
        r2_mask[list(r2_indices)] = True
        differences.append(values[r2_mask].mean() - values[~r2_mask].mean())
    differences = np.asarray(differences)
    p_value = float(np.mean(np.abs(differences) >= abs(observed) - 1e-15))
    return observed, p_value


def load_mapping(targets: tuple[str, ...]):
    paths = [
        MAPPING_DIR / f"mapping_stability_{animal.lower()}_{target}.csv"
        for animal in ANIMAL_SESSIONS for target in targets
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing mapping-stability results: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = sum(
        (len(r1) + len(r2)) * len(targets) * N_REPEATS * N_FOLDS
        for r1, r2 in ANIMAL_SESSIONS.values()
    )
    if len(frame) != expected:
        raise AssertionError(
            f"expected {expected} mapping rows, found {len(frame)}"
        )
    keys = ["animal", "target", "session", "repeat", "fold"]
    if frame.duplicated(keys).any():
        raise AssertionError("duplicate mapping-stability rows")
    sessions = frame.groupby(
        ["animal", "target", "epoch", "session"], as_index=False
    )[list(MAPPING_METRICS)].mean()
    rows = []
    for (animal, target), group in sessions.groupby(["animal", "target"]):
        for metric in MAPPING_METRICS:
            difference, p_value = exact_label_permutation(
                group[metric].to_numpy(), group.epoch.to_numpy()
            )
            rows.append({
                "animal": animal,
                "target": target,
                "metric": metric,
                "r1_mean": group.loc[group.epoch == "R1", metric].mean(),
                "r2_mean": group.loc[group.epoch == "R2", metric].mean(),
                "r2_minus_r1": difference,
                "exact_two_sided_p": p_value,
                "n_r1_sessions": int((group.epoch == "R1").sum()),
                "n_r2_sessions": int((group.epoch == "R2").sum()),
            })
    return frame, sessions, pd.DataFrame(rows)


def rank_correlation(first: pd.Series, second: pd.Series) -> float:
    return float(first.rank().corr(second.rank()))


def quality_relations(
    condition_pairs: pd.DataFrame,
    consensus_pairs: pd.DataFrame,
    mapping_sessions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_rows = []
    for keys, group in condition_pairs.groupby(
        ["animal", "target", "decoder", "variant"]
    ):
        condition_rows.append({
            "animal": keys[0], "target": keys[1],
            "decoder": keys[2], "variant": keys[3],
            "pearson_gap_vs_own_advantage": group.gap_common.corr(
                group.own_advantage_common
            ),
            "rank_gap_vs_own_advantage": rank_correlation(
                group.gap_common, group.own_advantage_common
            ),
            "raw_gap": group.gap_common.mean(),
            "excess_gap": group.excess_gap_common.mean(),
            "target_retention_gap": group.target_retention_gap_common.mean(),
            "retention_gap": group.retention_gap_common.mean(),
            "n_pairs": len(group),
        })

    wide = mapping_sessions.set_index(
        ["animal", "target", "session"]
    )[list(MAPPING_METRICS)]
    joined = consensus_pairs.copy()
    for metric in MAPPING_METRICS:
        r1_index = pd.MultiIndex.from_frame(
            joined[["animal", "target", "r1_session"]].rename(
                columns={"r1_session": "session"}
            )
        )
        r2_index = pd.MultiIndex.from_frame(
            joined[["animal", "target", "r2_session"]].rename(
                columns={"r2_session": "session"}
            )
        )
        joined[f"r1_{metric}"] = wide[metric].reindex(r1_index).to_numpy()
        joined[f"r2_{metric}"] = wide[metric].reindex(r2_index).to_numpy()
        joined[f"advantage_{metric}"] = (
            joined[f"r2_{metric}"] - joined[f"r1_{metric}"]
        )
    consensus_rows = []
    for (animal, target), group in joined.groupby(["animal", "target"]):
        for metric in MAPPING_METRICS:
            advantage = group[f"advantage_{metric}"]
            consensus_rows.append({
                "animal": animal, "target": target, "metric": metric,
                "pearson_consensus_gap_vs_advantage": group.consensus_median_gap.corr(
                    advantage
                ),
                "rank_consensus_gap_vs_advantage": rank_correlation(
                    group.consensus_median_gap, advantage
                ),
                "mean_quality_advantage": advantage.mean(),
                "n_pairs": len(group),
            })
    return pd.DataFrame(condition_rows), pd.DataFrame(consensus_rows), joined


def plot(
    condition_pairs: pd.DataFrame,
    condition_summary: pd.DataFrame,
    mapping_sessions: pd.DataFrame,
    consensus_joined: pd.DataFrame,
):
    target = "relative_position"
    labels = {
        ("ridge", "instantaneous"): "Ridge",
        ("wiener", "history_2"): "Wiener",
        ("arx", "trial_aware"): "ARX\ntrial",
        ("kalman", "original_concatenated"): "Kalman\nconcat",
        ("kalman", "trial_aware"): "Kalman\ntrial",
        ("kalman", "behaviour_center"): "Kalman\ncenter",
    }
    colors = ("#0072B2", "#56B4E9", "#E69F00", "#CC79A7", "#D55E00", "#009E73")
    rng = np.random.default_rng(20260808)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for ax, animal in zip(axes[0], ("TS", "TY"), strict=True):
        subset = condition_pairs[
            (condition_pairs.animal == animal) & (condition_pairs.target == target)
        ]
        summary = condition_summary[
            (condition_summary.animal == animal)
            & (condition_summary.target == target)
        ].set_index(["decoder", "variant"])
        means = [summary.at[condition, "gap_common"] for condition in CONDITION_ORDER]
        excess = [
            summary.at[condition, "excess_gap_common"]
            for condition in CONDITION_ORDER
        ]
        x = np.arange(len(CONDITION_ORDER))
        ax.bar(x, means, color=colors, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.scatter(
            x, excess, marker="D", facecolor="white", edgecolor="black",
            s=42, linewidth=1.0, zorder=5, label="target-difficulty adjusted",
        )
        for index, condition in enumerate(CONDITION_ORDER):
            values = subset[
                (subset.decoder == condition[0]) & (subset.variant == condition[1])
            ].gap_common.to_numpy()
            ax.scatter(
                index + rng.uniform(-0.10, 0.10, len(values)), values,
                color=colors[index], alpha=0.32, s=14, linewidths=0,
            )
        ax.axhline(0, color="0.25", linewidth=1)
        ax.set_xticks(x, [labels[c] for c in CONDITION_ORDER])
        ax.set_ylabel("R2→R1 − R1→R2 correlation")
        ax.set_title(f"{animal}: matched decoder gaps", loc="left", weight="bold")
        ax.grid(axis="y", alpha=0.2)
        if animal == "TY":
            ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[1, 0]
    mapping = mapping_sessions[mapping_sessions.target == target]
    positions = {("TS", "R1"): 0, ("TS", "R2"): 1, ("TY", "R1"): 3, ("TY", "R2"): 4}
    epoch_colors = {"R1": "#777777", "R2": "#E69F00"}
    for (animal, epoch), x in positions.items():
        values = mapping[
            (mapping.animal == animal) & (mapping.epoch == epoch)
        ].within_corr.to_numpy()
        ax.scatter(
            x + rng.uniform(-0.08, 0.08, len(values)), values,
            color=epoch_colors[epoch], s=28, alpha=0.75,
        )
        ax.hlines(values.mean(), x - 0.2, x + 0.2, color=epoch_colors[epoch], lw=3)
    ax.set_xticks([0, 1, 3, 4], ["TS R1", "TS R2", "TY R1", "TY R2"])
    ax.set_ylabel("Held-out within-day ridge correlation")
    ax.set_title("Source mapping quality (32/8 trials)", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    relation = consensus_joined[consensus_joined.target == target]
    for animal, color in (("TS", "#0072B2"), ("TY", "#D55E00")):
        subset = relation[relation.animal == animal]
        ax.scatter(
            subset.advantage_within_corr, subset.consensus_median_gap,
            color=color, label=animal, alpha=0.65, s=30,
        )
    ax.axhline(0, color="0.35", linewidth=1)
    ax.axvline(0, color="0.35", linewidth=1)
    ax.set_xlabel("R2 − R1 within-day mapping quality")
    ax.set_ylabel("Four-decoder median gap")
    ax.set_title("Does source quality predict transfer direction?", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.suptitle(
        "Decoder-independent component versus source-conditioned interaction",
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    targets = tuple(args.targets)
    raw = add_transfer_metrics(load_consensus(targets))
    condition_pairs, condition_day, condition_summary = condition_summaries(raw)
    consensus_split, consensus_pair, consensus_day, consensus_summary = (
        decoder_consensus(raw)
    )
    mapping_raw, mapping_sessions, mapping_summary = load_mapping(targets)
    condition_relation, consensus_relation, consensus_joined = quality_relations(
        condition_pairs, consensus_pair, mapping_sessions
    )
    outputs = {
        "decoder_consensus_all.csv": raw,
        "decoder_condition_pair_means.csv": condition_pairs,
        "decoder_condition_by_r2_session.csv": condition_day,
        "decoder_condition_summary.csv": condition_summary,
        "decoder_consensus_splits.csv": consensus_split,
        "decoder_consensus_pair_means.csv": consensus_pair,
        "decoder_consensus_by_r2_session.csv": consensus_day,
        "decoder_consensus_summary.csv": consensus_summary,
        "mapping_stability_all.csv": mapping_raw,
        "mapping_stability_session_means.csv": mapping_sessions,
        "mapping_stability_summary.csv": mapping_summary,
        "decoder_gap_source_quality_relation.csv": condition_relation,
        "consensus_gap_mapping_quality_relation.csv": consensus_relation,
        "consensus_pair_mapping_join.csv": consensus_joined,
    }
    for filename, frame in outputs.items():
        directory = MAPPING_DIR if filename.startswith("mapping_") else CONSENSUS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / filename, index=False)
    if "relative_position" in targets:
        plot(
            condition_pairs, condition_summary,
            mapping_sessions, consensus_joined,
        )
    print("DECODER CONDITIONS")
    print(condition_summary.round(4).to_string(index=False))
    print("\nFOUR-DECODER CONSENSUS")
    print(consensus_summary.round(4).to_string(index=False))
    print("\nMAPPING STABILITY")
    print(mapping_summary.round(4).to_string(index=False))
    print(f"\nsaved {FIGURE}")


if __name__ == "__main__":
    main()
