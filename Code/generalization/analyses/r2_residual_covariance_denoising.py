"""Cross-fitted R2 residual-covariance denoising toward R1.

Question
--------
R1 neural activity is quieter than R2 after position + velocity are regressed
out.  If the extra R2 variability is nuisance noise, does shrinking the R2
movement-residual covariance toward R1 improve R1->R2 decoding and reduce the
R2->R1 minus R1->R2 asymmetry?

For every R1/R2 session pair and trial-grouped fold, calibration trials alone
fit session PCA, single-trial CCA, kinematic signal models, residual
covariances, and the R2->R1 covariance target.  Calibration trials then train a
neural-only linear denoiser to approximate that target.  Held-out R2 trials are
transformed from neural activity alone: their true kinematics are used only for
post-hoc covariance diagnostics, never to construct decoder inputs.  No trials
are deleted.

Conditions
----------
raw
    Unmodified R2 canonical activity.
isotropic_trace_shrink
    Uniformly shrink R2 residuals until their calibration covariance trace
    equals R1's (increase is forbidden).
directional_shrink
    Match generalized covariance directions where R2 is noisier than R1, but
    never amplify directions where R2 is quieter.
full_covariance_match
    Exact regularized covariance transport from R2 residual covariance to R1;
    this may amplify a direction when R1 is broader there and is included as a
    covariance-matching sensitivity analysis rather than pure denoising.

Outputs
-------
Results/manifold_geometry/r2_residual_covariance_denoising_crossfit.csv
Results/manifold_geometry/r2_residual_covariance_denoising_pairs.csv
Results/manifold_geometry/r2_residual_covariance_denoising_r2_days.csv
Results/manifold_geometry/r2_residual_covariance_denoising_summary.csv
Results/manifold_geometry/figures/fig_r2_residual_covariance_denoising.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from big_sweep_phase2_crossday import (
    EXCLUDE_TRIALS,
    SESSIONS_R1,
    SESSIONS_R2,
    kalman_fit_predict,
    m2_per_trial,
)
from nested_cca_validation import load_session
from private_readout_crossfit import fit_calibration_alignment, trial_folds

warnings.filterwarnings("ignore")

K = 12
N_FOLDS = 5
N_REPEATS = 1
SEED = 20260722
TARGET = "relative_position"
RIDGE_FRACTION = 1e-3
DENOISER_L2 = 1e-3
CONDITIONS = (
    "raw",
    "isotropic_trace_shrink",
    "directional_shrink",
    "full_covariance_match",
)

REPO = _THIS.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry"
OUT_CROSSFIT = OUT_DIR / "r2_residual_covariance_denoising_crossfit.csv"
OUT_PAIRS = OUT_DIR / "r2_residual_covariance_denoising_pairs.csv"
OUT_DAYS = OUT_DIR / "r2_residual_covariance_denoising_r2_days.csv"
OUT_SUMMARY = OUT_DIR / "r2_residual_covariance_denoising_summary.csv"
FIG = OUT_DIR / "figures" / "fig_r2_residual_covariance_denoising.png"


def short_session(session: str) -> str:
    match = re.search(r"2025(\d{4})", session)
    return match.group(1) if match else session


def symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return (matrix + matrix.T) / 2.0


def spd_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """Symmetric matrix power for a numerically positive-definite matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric(matrix))
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    return (eigenvectors * (eigenvalues**power)) @ eigenvectors.T


def regularized_covariance(residual: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf covariance plus a small trace-scaled numerical ridge."""
    residual = np.asarray(residual, dtype=float)
    centered = residual - residual.mean(axis=0, keepdims=True)
    covariance = LedoitWolf(assume_centered=True).fit(centered).covariance_
    scale = float(np.trace(covariance) / covariance.shape[0])
    return symmetric(
        covariance + RIDGE_FRACTION * max(scale, 1e-12) * np.eye(covariance.shape[0])
    )


def covariance_transport(
    source_covariance: np.ndarray,
    target_covariance: np.ndarray,
    *,
    shrink_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Map row-vector source residuals toward the target covariance.

    If ``shrink_only`` is False, ``A.T @ source @ A == target`` up to numerical
    precision.  If true, generalized target/source eigenvalues above one are
    clipped to one so no source-whitened direction is amplified.
    """
    source = symmetric(source_covariance)
    target = symmetric(target_covariance)
    source_sqrt = spd_power(source, 0.5)
    source_inv_sqrt = spd_power(source, -0.5)
    relative_target = symmetric(source_inv_sqrt @ target @ source_inv_sqrt)
    eigenvalues, eigenvectors = np.linalg.eigh(relative_target)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    used = np.minimum(eigenvalues, 1.0) if shrink_only else eigenvalues
    relative_map = (eigenvectors * np.sqrt(used)) @ eigenvectors.T
    transform = source_inv_sqrt @ relative_map @ source_sqrt
    return transform, eigenvalues


def isotropic_trace_transform(
    source_covariance: np.ndarray, target_covariance: np.ndarray
) -> tuple[np.ndarray, float]:
    ratio = float(np.trace(target_covariance) / np.trace(source_covariance))
    scale = min(1.0, np.sqrt(max(ratio, 0.0)))
    return np.eye(source_covariance.shape[0]) * scale, scale


def relative_covariance_error(observed: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(observed - target, ord="fro")
        / max(np.linalg.norm(target, ord="fro"), 1e-12)
    )


def kinematic_design(position: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Position + per-trial finite-difference velocity."""
    position = np.asarray(position, dtype=float)
    velocity = np.zeros_like(position)
    for indices in meta.groupby("trial_number").indices.values():
        indices = np.asarray(indices)
        if len(indices) > 1:
            velocity[indices] = np.gradient(position[indices], 0.030, axis=0)
    return np.column_stack([position, velocity])


def fit_kinematic_signal(
    activity: np.ndarray,
    kinematics: np.ndarray,
    calibration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit affine kinematics->activity on calibration rows and predict all rows."""
    mean = kinematics[calibration].mean(axis=0)
    scale = kinematics[calibration].std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (kinematics - mean) / scale
    design = np.column_stack([standardized, np.ones(len(standardized))])
    coefficients, *_ = np.linalg.lstsq(
        design[calibration], activity[calibration], rcond=None
    )
    signal = design @ coefficients
    return signal, np.asarray(activity, dtype=float) - signal


def fit_neural_only_denoiser(
    activity: np.ndarray,
    calibration_target: np.ndarray,
    calibration: np.ndarray,
) -> np.ndarray:
    """Learn calibration raw-neural -> quiet-neural and transform all rows.

    ``calibration_target`` has one row per True value in ``calibration``.  The
    returned held-out rows depend only on their neural activity and calibration
    parameters, not on held-out kinematics.
    """
    source = np.asarray(activity[calibration], dtype=float)
    target = np.asarray(calibration_target, dtype=float)
    if source.shape != target.shape:
        raise ValueError("calibration source and quiet target must share a shape")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    gram = source_centered.T @ source_centered
    ridge_scale = float(np.trace(gram) / gram.shape[0])
    coefficients = np.linalg.solve(
        gram + DENOISER_L2 * max(ridge_scale, 1e-12) * np.eye(gram.shape[0]),
        source_centered.T @ target_centered,
    )
    return (np.asarray(activity, dtype=float) - source_mean) @ coefficients + target_mean


def decode(
    source: dict,
    source_activity: np.ndarray,
    source_mask: np.ndarray,
    target: dict,
    target_activity: np.ndarray,
    target_mask: np.ndarray,
) -> float:
    target_meta = target["meta"][target_mask].reset_index(drop=True)
    state, prediction = kalman_fit_predict(
        source["X"][source_mask],
        source_activity[source_mask],
        target["X"][target_mask],
        target_activity[target_mask],
        target_meta,
    )
    return m2_per_trial(state, prediction, target_meta)


def exact_sign_flip_p(values) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    signs = np.asarray(list(product([-1.0, 1.0], repeat=len(values))))
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= observed - 1e-12))


def evaluate_fold(
    r1: dict,
    r2: dict,
    calibration_r1: np.ndarray,
    calibration_r2: np.ndarray,
    evaluation_r1: np.ndarray,
    evaluation_r2: np.ndarray,
    fit_seed: int,
) -> list[dict]:
    alignment = fit_calibration_alignment(
        r1, r2, calibration_r1, calibration_r2, fit_seed
    )
    z1 = alignment.transform_train(r1["Y"])
    z2 = alignment.transform_target(r2["Y"])
    signal1, residual1 = fit_kinematic_signal(z1, r1["Kin"], calibration_r1)
    signal2, residual2 = fit_kinematic_signal(z2, r2["Kin"], calibration_r2)
    covariance1 = regularized_covariance(residual1[calibration_r1])
    covariance2 = regularized_covariance(residual2[calibration_r2])

    isotropic, isotropic_scale = isotropic_trace_transform(covariance2, covariance1)
    directional, generalized_eigenvalues = covariance_transport(
        covariance2, covariance1, shrink_only=True
    )
    full, _ = covariance_transport(covariance2, covariance1, shrink_only=False)
    transforms = {
        "raw": np.eye(K),
        "isotropic_trace_shrink": isotropic,
        "directional_shrink": directional,
        "full_covariance_match": full,
    }

    own_r1 = decode(r1, z1, calibration_r1, r1, z1, evaluation_r1)
    evaluation_covariance1 = regularized_covariance(residual1[evaluation_r1])
    rows = []
    for condition, transform in transforms.items():
        if condition == "raw":
            quiet2 = z2
        else:
            oracle_calibration_target = (
                signal2[calibration_r2]
                + residual2[calibration_r2] @ transform
            )
            quiet2 = fit_neural_only_denoiser(
                z2, oracle_calibration_target, calibration_r2
            )
        # True kinematics enter only these post-hoc diagnostics.  ``quiet2`` was
        # already fixed above from held-out neural activity alone.
        transformed_calibration_residual = (
            quiet2[calibration_r2] - signal2[calibration_r2]
        )
        transformed_calibration_covariance = regularized_covariance(
            transformed_calibration_residual
        )
        transformed_evaluation_residual = quiet2[evaluation_r2] - signal2[evaluation_r2]
        transformed_evaluation_covariance = regularized_covariance(
            transformed_evaluation_residual
        )
        forward = decode(r1, z1, calibration_r1, r2, quiet2, evaluation_r2)
        reverse = decode(r2, quiet2, calibration_r2, r1, z1, evaluation_r1)
        own_r2 = decode(r2, quiet2, calibration_r2, r2, quiet2, evaluation_r2)
        rows.append({
            "condition": condition,
            "forward_corr": forward,
            "reverse_corr": reverse,
            "gap": reverse - forward,
            "own_r1_corr": own_r1,
            "own_r2_corr": own_r2,
            "r1_calibration_residual_trace": float(np.trace(covariance1)),
            "r2_calibration_residual_trace": float(np.trace(covariance2)),
            "r2_transformed_calibration_trace": float(
                np.trace(transformed_calibration_covariance)
            ),
            "calibration_trace_ratio": float(
                np.trace(transformed_calibration_covariance) / np.trace(covariance1)
            ),
            "calibration_covariance_error": relative_covariance_error(
                transformed_calibration_covariance, covariance1
            ),
            "r1_evaluation_residual_trace": float(
                np.trace(evaluation_covariance1)
            ),
            "r2_transformed_evaluation_trace": float(
                np.trace(transformed_evaluation_covariance)
            ),
            "evaluation_trace_ratio": float(
                np.trace(transformed_evaluation_covariance)
                / np.trace(evaluation_covariance1)
            ),
            "evaluation_covariance_error": relative_covariance_error(
                transformed_evaluation_covariance, evaluation_covariance1
            ),
            "isotropic_scale": isotropic_scale,
            "n_generalized_directions_shrunk": int(
                np.sum(generalized_eigenvalues < 1.0)
            ),
            "generalized_eigenvalue_min": float(generalized_eigenvalues.min()),
            "generalized_eigenvalue_max": float(generalized_eigenvalues.max()),
        })
    return rows


def summarize(result: pd.DataFrame):
    metric_columns = [
        "forward_corr",
        "reverse_corr",
        "gap",
        "own_r1_corr",
        "own_r2_corr",
        "calibration_trace_ratio",
        "calibration_covariance_error",
        "evaluation_trace_ratio",
        "evaluation_covariance_error",
    ]
    pairs = (
        result.groupby(["r1", "r2", "condition"], as_index=False)[metric_columns]
        .mean()
    )
    days = pairs.groupby(["r2", "condition"], as_index=False)[metric_columns].mean()

    rows = []
    raw_days = days[days.condition == "raw"].set_index("r2")
    for condition in CONDITIONS:
        condition_pairs = pairs[pairs.condition == condition]
        condition_days = days[days.condition == condition].set_index("r2")
        row = {"condition": condition, "n_pairs": len(condition_pairs)}
        for metric in metric_columns:
            values = condition_pairs[metric].to_numpy()
            row[metric] = float(values.mean())
            row[f"{metric}_sd_pairs"] = float(values.std(ddof=1))
        for metric in ("forward_corr", "reverse_corr", "gap", "own_r2_corr"):
            deltas = condition_days[metric] - raw_days[metric]
            row[f"delta_{metric}_vs_raw"] = float(deltas.mean())
            row[f"delta_{metric}_r2_day_signflip_p"] = (
                np.nan if condition == "raw" else exact_sign_flip_p(deltas)
            )
        rows.append(row)
    return pairs, days, pd.DataFrame(rows)


def make_figure(days: pd.DataFrame):
    labels = ["raw", "trace\nshrink", "directional\nshrink", "full cov\nmatch"]
    colors = ["#4d4d4d", "#e67e22", "#2ca02c", "#3498db"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))

    def day_lines(ax, metric):
        pivot = days.pivot(index="r2", columns="condition", values=metric)[list(CONDITIONS)]
        for _, values in pivot.iterrows():
            ax.plot(range(len(CONDITIONS)), values.to_numpy(), color="0.78", lw=1)
        for index, condition in enumerate(CONDITIONS):
            values = pivot[condition].to_numpy()
            sem = values.std(ddof=1) / np.sqrt(len(values))
            ax.errorbar(index, values.mean(), yerr=sem, fmt="o", color=colors[index],
                        ms=8, capsize=4, zorder=3)

    day_lines(axes[0], "evaluation_trace_ratio")
    axes[0].axhline(1, color="black", ls="--", lw=1)
    axes[0].set_ylabel("held-out residual trace: R2 / R1")
    axes[0].set_title("A  Does denoising generalize?")

    for metric, marker, label in [
        ("forward_corr", "o", "R1→R2"),
        ("reverse_corr", "s", "R2→R1"),
    ]:
        pivot = days.groupby("condition")[metric].mean().reindex(CONDITIONS)
        axes[1].plot(range(len(CONDITIONS)), pivot, marker=marker, label=label)
    axes[1].set_ylabel("held-out decode correlation")
    axes[1].set_title("B  Directional decoding")
    axes[1].legend(frameon=False)

    day_lines(axes[2], "gap")
    axes[2].axhline(0, color="black", lw=1)
    axes[2].set_ylabel("R2→R1 − R1→R2")
    axes[2].set_title("C  Decode asymmetry")

    day_lines(axes[3], "own_r2_corr")
    axes[3].set_ylabel("R2 within-day correlation")
    axes[3].set_title("D  Preserve R2 information?")

    for ax in axes:
        ax.set_xticks(range(len(CONDITIONS)), labels)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Cross-fitted R2 movement-residual covariance denoising", fontsize=14)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs_to_run = list(product(SESSIONS_R1, SESSIONS_R2))
    if args.max_pairs is not None:
        pairs_to_run = pairs_to_run[: args.max_pairs]
    sessions = list(dict.fromkeys(
        [session for pair in pairs_to_run for session in pair]
    ))
    print(f"Loading {len(sessions)} sessions ...", flush=True)
    cache = {
        session: load_session(session, TARGET, EXCLUDE_TRIALS.get(session, ()))
        for session in sessions
    }
    for data in cache.values():
        data["Kin"] = kinematic_design(data["X"], data["meta"])

    rows = []
    for pair_index, (r1_session, r2_session) in enumerate(pairs_to_run):
        r1, r2 = cache[r1_session], cache[r2_session]
        for repeat in range(args.repeats):
            split_seed = SEED + pair_index * 10_000 + repeat * 100
            folds1 = trial_folds(r1["meta"], N_FOLDS, split_seed)
            folds2 = trial_folds(r2["meta"], N_FOLDS, split_seed + 1)
            for fold in range(N_FOLDS):
                evaluation1, evaluation2 = folds1[fold], folds2[fold]
                calibration1, calibration2 = ~evaluation1, ~evaluation2
                fold_rows = evaluate_fold(
                    r1,
                    r2,
                    calibration1,
                    calibration2,
                    evaluation1,
                    evaluation2,
                    split_seed + fold,
                )
                for row in fold_rows:
                    row.update({
                        "r1": short_session(r1_session),
                        "r2": short_session(r2_session),
                        "r1_session": r1_session,
                        "r2_session": r2_session,
                        "repeat": repeat,
                        "fold": fold,
                        "n_calibration_r1_trials": int(
                            r1["meta"][calibration1]["trial_number"].nunique()
                        ),
                        "n_calibration_r2_trials": int(
                            r2["meta"][calibration2]["trial_number"].nunique()
                        ),
                        "n_evaluation_r1_trials": int(
                            r1["meta"][evaluation1]["trial_number"].nunique()
                        ),
                        "n_evaluation_r2_trials": int(
                            r2["meta"][evaluation2]["trial_number"].nunique()
                        ),
                    })
                    rows.append(row)
        print(
            f"[{pair_index + 1}/{len(pairs_to_run)}] "
            f"{short_session(r1_session)} vs {short_session(r2_session)}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CROSSFIT, index=False)
    pairs, days, summary = summarize(result)
    pairs.to_csv(OUT_PAIRS, index=False)
    days.to_csv(OUT_DAYS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    make_figure(days)

    columns = [
        "condition",
        "evaluation_trace_ratio",
        "evaluation_covariance_error",
        "forward_corr",
        "reverse_corr",
        "gap",
        "own_r2_corr",
        "delta_gap_vs_raw",
        "delta_gap_r2_day_signflip_p",
    ]
    n_session_pairs = result[["r1", "r2"]].drop_duplicates().shape[0]
    print(f"\n=== Cross-fitted summary ({n_session_pairs} session pairs) ===")
    print(summary[columns].round(3).to_string(index=False))
    print(f"\nSaved {OUT_CROSSFIT}")
    print(f"Saved {OUT_PAIRS}")
    print(f"Saved {OUT_DAYS}")
    print(f"Saved {OUT_SUMMARY}")
    print(f"Saved {FIG}")


if __name__ == "__main__":
    main()
