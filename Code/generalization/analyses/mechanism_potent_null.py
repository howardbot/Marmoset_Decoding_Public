"""Cross-fitted, dimension-balanced output-potent/output-null analysis.

This is the Kaufman-style analysis adapted to these data:

* 9-D kinematics = relative xyz position, velocity, and acceleration.
* Calibration-only standardization and PCA reduce kinematics from 9 to 6 D.
* Calibration-only PCA reduces neural activity to 12 D.
* A ridge readout maps 12-D neural activity to 6-D kinematics.
* Its six-dimensional column space (equivalently, the row space in Kaufman's
  column-vector convention) is output-potent; the orthogonal complement is a
  six-dimensional output-null space.
* Equal dimensionality makes held-out squared-Frobenius energies directly
  comparable, yielding a balanced null/potent activity ratio.

R1 and R2 are different recording epochs, not preparation and movement periods
within the same trial.  Therefore this script does NOT call its result Kaufman's
prep/move-normalized tuning ratio.  It reports the unnormalized, balanced
null/potent energy ratio separately for R1 and R2, then compares sessions.

Every learned object (both PCAs, kinematic scaling, ridge alpha, readout, and
potent/null bases) is fit without the outer held-out trials.  Folds are grouped
by complete reaches.  Ridge alpha is selected by an inner trial-grouped CV.

Outputs are written below Results/manifold_geometry/balanced_potent_null/.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

# Keep each session worker single-threaded; parallelism is across sessions.
for _variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS,
    EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS,
    TRIAL_RESULTS,
    UNIT_QUALITIES,
    filter_trials,
)
from readout_subspaces import orthogonal_complement, readout_basis

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - the HatLab environment includes it
    threadpool_limits = None


BIN_MS = 30
NEURAL_DIM = 12
KINEMATIC_RAW_DIM = 9
KINEMATIC_PCA_DIM = 6
TARGET = "relative_position_velocity_acceleration"
SMOOTHER_KW = {
    "smoother": "butter",
    "smooth_cutoff_hz": 6.0,
    "smooth_order": 2,
}
N_FOLDS = 5
N_REPEATS = 5
N_INNER_FOLDS = 4
SEED = 20260803
RIDGE_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

REPO_ROOT = _THIS.parents[2]
DEFAULT_OUT_DIR = (
    REPO_ROOT / "Results" / "manifold_geometry" / "balanced_potent_null"
)

# Keep training mean and std
@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

#  The PCA model,
@dataclass(frozen=True)
class PCAModel:
    mean: np.ndarray
    components: np.ndarray
    explained_fraction: float
    component_variance: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) @ self.components

# Readout model
@dataclass(frozen=True)
class RidgeModel:
    weights: np.ndarray
    intercept: np.ndarray
    alpha: float

    def predict(self, activity: np.ndarray) -> np.ndarray:
        return np.asarray(activity, dtype=float) @ self.weights + self.intercept

# fit the model
def fit_standardizer(values: np.ndarray) -> Standardizer:
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return Standardizer(mean=mean, scale=scale)

# No sklearn, SVD manually
def fit_pca(values: np.ndarray, n_components: int) -> PCAModel:
    """Fit an orthogonal PCA using calibration rows only."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("PCA values must be a 2-D array with at least two rows")
    if not 1 <= n_components <= min(values.shape):
        raise ValueError(
            f"Cannot fit {n_components} PCs to an array with shape {values.shape}"
        )
    mean = values.mean(axis=0)
    centered = values - mean
    _, singular_values, right_t = np.linalg.svd(centered, full_matrices=False)
    total_ss = float(np.sum(singular_values ** 2))
    kept_ss = float(np.sum(singular_values[:n_components] ** 2))
    variance = singular_values[:n_components] ** 2 / max(len(values) - 1, 1)
    return PCAModel(
        mean=mean,
        components=right_t[:n_components].T,
        explained_fraction=kept_ss / total_ss if total_ss > 0 else np.nan,
        component_variance=variance,
    )

# preparing trials for n-fold training
def grouped_trial_folds(meta: pd.DataFrame, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return evaluation masks whose units are complete trials."""
    trials = np.asarray(sorted(meta["trial_number"].unique()))
    if n_folds < 2:
        raise ValueError("At least two outer folds are required")
    if len(trials) < 2 * n_folds:
        raise ValueError(
            f"Need at least {2 * n_folds} trials for {n_folds}-fold CV; "
            f"found {len(trials)}"
        )
    shuffled = np.random.default_rng(seed).permutation(trials)
    groups = np.array_split(shuffled, n_folds)
    return [meta["trial_number"].isin(group).to_numpy() for group in groups]

# Fit the ridge readout
def fit_ridge(activity: np.ndarray, movement: np.ndarray, alpha: float) -> RidgeModel:
    """Fit ridge with an intercept and a dimensionless, trace-scaled alpha."""
    activity = np.asarray(activity, dtype=float)
    movement = np.asarray(movement, dtype=float)
    activity_mean = activity.mean(axis=0)
    movement_mean = movement.mean(axis=0)
    z = activity - activity_mean
    x = movement - movement_mean
    gram = z.T @ z
    scale = float(np.trace(gram) / max(activity.shape[1], 1))
    penalty = float(alpha) * max(scale, 1e-12)
    system = gram + penalty * np.eye(activity.shape[1])
    rhs = z.T @ x
    try:
        weights = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, rhs, rcond=None)[0]
    intercept = movement_mean - activity_mean @ weights
    return RidgeModel(weights=weights, intercept=intercept, alpha=float(alpha))

# Loop the best alpha value
def select_ridge_alpha(
    activity: np.ndarray,
    movement: np.ndarray,
    meta: pd.DataFrame,
    alpha_grid: tuple[float, ...],
    n_inner_folds: int,
    seed: int,
) -> tuple[float, float]:
    """Select alpha on outer-calibration trials only."""
    n_trials = int(meta["trial_number"].nunique())
    inner_folds = min(n_inner_folds, max(2, n_trials // 2))
    masks = grouped_trial_folds(meta, inner_folds, seed)
    mean_mse = []
    for alpha in alpha_grid:
        errors = []
        for validation in masks:
            model = fit_ridge(activity[~validation], movement[~validation], alpha)
            residual = movement[validation] - model.predict(activity[validation])
            errors.append(float(np.mean(residual ** 2)))
        mean_mse.append(float(np.mean(errors)))
    best_index = int(np.nanargmin(mean_mse))
    return float(alpha_grid[best_index]), float(mean_mse[best_index])

# calculating R2
def multivariate_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual_ss = float(np.sum((observed - predicted) ** 2))
    total_ss = float(np.sum((observed - observed.mean(axis=0)) ** 2))
    return 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan

# Also calculating the corr
def mean_dimension_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    correlations = []
    for dimension in range(observed.shape[1]):
        a = observed[:, dimension]
        b = predicted[:, dimension]
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            continue
        correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(correlations)) if correlations else np.nan

# split the training and testing trials
def evaluate_split(
    data: dict,
    calibration: np.ndarray,
    evaluation: np.ndarray,
    *,
    alpha_grid: tuple[float, ...],
    n_inner_folds: int,
    seed: int,
) -> dict:
    """Fit all transforms on calibration and evaluate one held-out fold."""
    raw_movement = data["movement"]
    raw_neural = data["neural"]
    meta = data["meta"]

    neural_pca = fit_pca(raw_neural[calibration], NEURAL_DIM)
    neural = neural_pca.transform(raw_neural)

    # Position, velocity, and acceleration have different physical units.  Scale
    # the nine raw variables on calibration data before asking PCA for six axes.
    movement_scaler = fit_standardizer(raw_movement[calibration])
    movement_standardized = movement_scaler.transform(raw_movement)
    movement_pca = fit_pca(
        movement_standardized[calibration], KINEMATIC_PCA_DIM
    )
    movement = movement_pca.transform(movement_standardized)

    calibration_meta = meta.loc[calibration].reset_index(drop=True)
    selected_alpha, inner_cv_mse = select_ridge_alpha(
        neural[calibration],
        movement[calibration],
        calibration_meta,
        alpha_grid,
        n_inner_folds,
        seed,
    )
    model = fit_ridge(
        neural[calibration], movement[calibration], selected_alpha
    )

    # Row-oriented data use movement = neural @ weights.  Therefore weights is
    # 12 x 6, and its COLUMN space is the neural potent space.  This is exactly
    # the row space of Kaufman's 6 x 12 W in column-vector notation.
    potent_basis = readout_basis(model.weights, tolerance=1e-10)
    null_basis = orthogonal_complement(potent_basis, NEURAL_DIM)
    rank = int(potent_basis.shape[1])
    null_dimension = int(null_basis.shape[1])
    potent_projector = potent_basis @ potent_basis.T
    null_projector = null_basis @ null_basis.T

    evaluation_neural = neural[evaluation]
    evaluation_movement = movement[evaluation]
    centered_evaluation = evaluation_neural - evaluation_neural.mean(axis=0)
    potent_coordinates = centered_evaluation @ potent_basis
    null_coordinates = centered_evaluation @ null_basis
    potent_energy = float(np.sum(potent_coordinates ** 2))
    null_energy = float(np.sum(null_coordinates ** 2))
    total_energy = float(np.sum(centered_evaluation ** 2))
    balanced = rank == KINEMATIC_PCA_DIM and null_dimension == KINEMATIC_PCA_DIM
    ratio = null_energy / potent_energy if balanced and potent_energy > 0 else np.nan

    prediction = model.predict(evaluation_neural)
    fixed_potent_prediction = (
        evaluation_neural @ potent_projector @ model.weights + model.intercept
    )
    fixed_null_output = evaluation_neural @ null_projector @ model.weights
    # Refitting the decoder!
    # Optional empirical controls: refit equal-rank decoders using coordinates
    # from either subspace.  These are distinct from the defining fixed readout.
    potent_refit = fit_ridge(
        neural[calibration] @ potent_basis,
        movement[calibration],
        selected_alpha,
    )
    null_refit = fit_ridge(
        neural[calibration] @ null_basis,
        movement[calibration],
        selected_alpha,
    )
    potent_refit_prediction = potent_refit.predict(
        evaluation_neural @ potent_basis
    )
    null_refit_prediction = null_refit.predict(evaluation_neural @ null_basis)

    singular_values = np.linalg.svd(model.weights, compute_uv=False)
    identity = np.eye(NEURAL_DIM)
    return {
        "n_calibration_samples": int(calibration.sum()),
        "n_evaluation_samples": int(evaluation.sum()),
        "n_calibration_trials": int(meta.loc[calibration, "trial_number"].nunique()),
        "n_evaluation_trials": int(meta.loc[evaluation, "trial_number"].nunique()),
        "neural_dimension": NEURAL_DIM,
        "kinematic_raw_dimension": KINEMATIC_RAW_DIM,
        "kinematic_pca_dimension": KINEMATIC_PCA_DIM,
        "potent_dimension": rank,
        "null_dimension": null_dimension,
        "dimension_balanced": bool(balanced),
        "neural_pca_explained": float(neural_pca.explained_fraction),
        "kinematic_pca_explained": float(movement_pca.explained_fraction),
        "ridge_alpha": selected_alpha,
        "inner_cv_mse": inner_cv_mse,
        "readout_r2": multivariate_r2(evaluation_movement, prediction),
        "readout_mean_pc_corr": mean_dimension_correlation(
            evaluation_movement, prediction
        ),
        "fixed_potent_r2": multivariate_r2(
            evaluation_movement, fixed_potent_prediction
        ),
        "potent_refit_r2": multivariate_r2(
            evaluation_movement, potent_refit_prediction
        ),
        "null_refit_r2": multivariate_r2(
            evaluation_movement, null_refit_prediction
        ),
        "potent_energy": potent_energy,
        "null_energy": null_energy,
        "total_centered_energy": total_energy,
        "potent_energy_per_coordinate": potent_energy / max(
            int(evaluation.sum()) * rank, 1
        ),
        "null_energy_per_coordinate": null_energy / max(
            int(evaluation.sum()) * null_dimension, 1
        ),
        "balanced_null_potent_ratio": ratio,
        "log_balanced_ratio": float(np.log(ratio)) if ratio > 0 else np.nan,
        "potent_energy_fraction": potent_energy / total_energy,
        "null_energy_fraction": null_energy / total_energy,
        "readout_singular_max": float(singular_values[0]),
        "readout_singular_min": float(singular_values[-1]),
        "readout_condition": float(singular_values[0] / singular_values[-1]),
        # Numerical proofs of the decomposition.
        "max_potent_null_dot": float(
            np.max(np.abs(potent_basis.T @ null_basis))
        ),
        "projector_completeness_error": float(
            np.max(np.abs(potent_projector + null_projector - identity))
        ),
        "activity_reconstruction_relative_error": float(
            np.linalg.norm(
                centered_evaluation
                - centered_evaluation @ potent_projector
                - centered_evaluation @ null_projector
            ) / max(np.linalg.norm(centered_evaluation), 1e-12)
        ),
        "potent_readout_relative_error": float(
            np.linalg.norm(potent_projector @ model.weights - model.weights)
            / max(np.linalg.norm(model.weights), 1e-12)
        ),
        "null_fixed_output_relative": float(
            np.linalg.norm(fixed_null_output)
            / max(np.linalg.norm(evaluation_neural @ model.weights), 1e-12)
        ),
        "fixed_potent_prediction_relative_error": float(
            np.linalg.norm(fixed_potent_prediction - prediction)
            / max(np.linalg.norm(prediction), 1e-12)
        ),
    }


def load_session(session: str) -> dict:
    """Build the 9-D movement and smoothed neural matrices for one session."""
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    if not du.PROCESSED_NWB.exists():
        raise FileNotFoundError(du.PROCESSED_NWB)
    # Decoder utilities print detailed per-session diagnostics.  Workers return a
    # compact progress record instead, so parallel logs remain readable.
    with contextlib.redirect_stdout(io.StringIO()):
        nwb_io, nwb, reaches = du.load_nwb_and_reach()
        try:
            movement, neural, meta = du.build_decoder_dataset(
                nwb,
                reaches,
                TARGET,
                bin_size=BIN_MS / 1000.0,
                unit_qualities=UNIT_QUALITIES,
                trial_results=TRIAL_RESULTS,
                trial_window="start_to_peak",
                **SMOOTHER_KW,
            )
        finally:
            nwb_io.close()
    movement, neural, meta = filter_trials(
        movement, neural, meta, EXCLUDE_TRIALS.get(session, ())
    )
    neural = du.smooth_neural_causal(
        neural, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS
    )
    if movement.shape[1] != KINEMATIC_RAW_DIM:
        raise RuntimeError(
            f"Expected 9-D position/velocity/acceleration; got {movement.shape}"
        )
    if neural.shape[1] < NEURAL_DIM:
        raise RuntimeError(
            f"Need at least {NEURAL_DIM} units; got {neural.shape[1]} in {session}"
        )
    return {"movement": movement, "neural": neural, "meta": meta}


def analyze_session(task: dict) -> dict:
    """Process one session; suitable for a spawned worker."""
    started = time.perf_counter()
    limiter = threadpool_limits(limits=1) if threadpool_limits else contextlib.nullcontext()
    with limiter:
        data = load_session(task["session"])
        rows = []
        for repeat in range(task["repeats"]):
            fold_seed = task["seed"] + task["session_index"] * 100_000 + repeat * 1_000
            evaluation_folds = grouped_trial_folds(
                data["meta"], task["folds"], fold_seed
            )
            for fold, evaluation in enumerate(evaluation_folds):
                result = evaluate_split(
                    data,
                    ~evaluation,
                    evaluation,
                    alpha_grid=task["alpha_grid"],
                    n_inner_folds=task["inner_folds"],
                    seed=fold_seed + fold + 1,
                )
                result.update({
                    "animal": task["animal"],
                    "epoch": task["epoch"],
                    "session": task["session"],
                    "session_date": task["session"][4:12],
                    "repeat": repeat,
                    "fold": fold,
                })
                rows.append(result)
    return {
        "session": task["session"],
        "epoch": task["epoch"],
        "rows": rows,
        "n_samples": len(data["movement"]),
        "n_trials": int(data["meta"]["trial_number"].nunique()),
        "n_units": int(data["neural"].shape[1]),
        "elapsed_seconds": time.perf_counter() - started,
    }


def aggregate_rows(split_df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate by summing held-out energies, then forming their ratio."""
    records = []
    diagnostic_columns = [
        "neural_pca_explained", "kinematic_pca_explained", "ridge_alpha",
        "inner_cv_mse", "readout_r2", "readout_mean_pc_corr",
        "fixed_potent_r2", "potent_refit_r2", "null_refit_r2",
        "potent_dimension", "null_dimension", "potent_energy_fraction",
        "null_energy_fraction", "readout_condition", "max_potent_null_dot",
        "projector_completeness_error", "activity_reconstruction_relative_error",
        "potent_readout_relative_error", "null_fixed_output_relative",
        "fixed_potent_prediction_relative_error",
    ]
    for keys, group in split_df.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        potent = float(group["potent_energy"].sum())
        null = float(group["null_energy"].sum())
        balanced = bool(group["dimension_balanced"].all())
        ratio = null / potent if balanced and potent > 0 else np.nan
        record.update({
            "n_splits": int(len(group)),
            "n_evaluation_samples_summed": int(group["n_evaluation_samples"].sum()),
            "potent_energy": potent,
            "null_energy": null,
            "dimension_balanced_all_splits": balanced,
            "balanced_null_potent_ratio": ratio,
            "log_balanced_ratio": float(np.log(ratio)) if ratio > 0 else np.nan,
        })
        for column in diagnostic_columns:
            record[f"mean_{column}"] = float(group[column].mean())
        records.append(record)
    return pd.DataFrame(records)


def exact_epoch_permutation(session_df: pd.DataFrame) -> dict:
    """Exact session-label permutation test of the R2-R1 mean log ratio."""
    r1 = session_df.loc[session_df["epoch"] == "R1", "log_balanced_ratio"].to_numpy()
    r2 = session_df.loc[session_df["epoch"] == "R2", "log_balanced_ratio"].to_numpy()
    if len(r1) == 0 or len(r2) == 0:
        return {}
    values = np.concatenate([r1, r2])
    observed = float(r2.mean() - r1.mean())
    n_r2 = len(r2)
    deltas = []
    all_indices = np.arange(len(values))
    for selected_tuple in combinations(range(len(values)), n_r2):
        selected = np.asarray(selected_tuple, dtype=int)
        mask = np.zeros(len(values), dtype=bool)
        mask[selected] = True
        deltas.append(float(values[mask].mean() - values[~mask].mean()))
    deltas = np.asarray(deltas)
    tolerance = 1e-12
    return {
        "metric": "session_log_balanced_null_potent_ratio",
        "n_r1_sessions": int(len(r1)),
        "n_r2_sessions": int(len(r2)),
        "mean_r1_log_ratio": float(r1.mean()),
        "mean_r2_log_ratio": float(r2.mean()),
        "delta_r2_minus_r1_log_ratio": observed,
        "ratio_of_geometric_means_r2_over_r1": float(np.exp(observed)),
        "exact_permutations": int(len(deltas)),
        "exact_two_sided_p": float(
            np.mean(np.abs(deltas) >= abs(observed) - tolerance)
        ),
        "exact_one_sided_r2_greater_p": float(
            np.mean(deltas >= observed - tolerance)
        ),
    }


def bootstrap_epoch_difference(
    session_df: pd.DataFrame, seed: int, n_bootstrap: int = 20_000
) -> dict:
    r1 = session_df.loc[session_df["epoch"] == "R1", "log_balanced_ratio"].to_numpy()
    r2 = session_df.loc[session_df["epoch"] == "R2", "log_balanced_ratio"].to_numpy()
    if len(r1) == 0 or len(r2) == 0:
        return {}
    rng = np.random.default_rng(seed)
    differences = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        differences[index] = (
            rng.choice(r2, size=len(r2), replace=True).mean()
            - rng.choice(r1, size=len(r1), replace=True).mean()
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "bootstrap_replicates": int(n_bootstrap),
        "bootstrap_log_delta_ci_low": float(low),
        "bootstrap_log_delta_ci_high": float(high),
        "bootstrap_ratio_of_geometric_means_ci_low": float(np.exp(low)),
        "bootstrap_ratio_of_geometric_means_ci_high": float(np.exp(high)),
    }


def make_summary_figure(session_df: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = session_df.sort_values(["epoch", "session_date"]).reset_index(drop=True)
    colors = ordered["epoch"].map({"R1": "#377eb8", "R2": "#e41a1c"})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].scatter(
        np.arange(len(ordered)),
        ordered["balanced_null_potent_ratio"],
        c=colors,
        s=48,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(len(ordered)))
    axes[0].set_xticklabels(ordered["session_date"], rotation=65, ha="right", fontsize=7)
    axes[0].set_ylabel("held-out null / potent energy")
    axes[0].set_title("Balanced 6D / 6D ratio by session")

    groups = [
        ordered.loc[ordered["epoch"] == epoch, "log_balanced_ratio"].to_numpy()
        for epoch in ("R1", "R2")
    ]
    for position, (epoch, values, color) in enumerate(
        zip(("R1", "R2"), groups, ("#377eb8", "#e41a1c"))
    ):
        jitter = np.linspace(-0.07, 0.07, len(values)) if len(values) > 1 else [0]
        axes[1].scatter(
            position + np.asarray(jitter), values, color=color, s=55,
            edgecolor="white", linewidth=0.6, label=epoch,
        )
        axes[1].plot(
            [position - 0.18, position + 0.18], [values.mean(), values.mean()],
            color="black", linewidth=2,
        )
    axes[1].axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    axes[1].set_xticks([0, 1], ["R1", "R2"])
    axes[1].set_ylabel("log(null / potent energy)")
    axes[1].set_title("Epoch comparison (sessions are the units)")
    figure.suptitle("12D neural → 6D position/velocity/acceleration readout")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_alpha_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("alpha grid must contain nonnegative numbers")
    return tuple(sorted(set(values)))


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    activity = rng.standard_normal((500, NEURAL_DIM))
    true_weights = rng.standard_normal((NEURAL_DIM, KINEMATIC_PCA_DIM))
    movement = activity @ true_weights
    model = fit_ridge(activity, movement, 0.0)
    potent = readout_basis(model.weights, tolerance=1e-10)
    null = orthogonal_complement(potent, NEURAL_DIM)
    assert potent.shape == (NEURAL_DIM, KINEMATIC_PCA_DIM)
    assert null.shape == (NEURAL_DIM, KINEMATIC_PCA_DIM)
    assert np.max(np.abs(potent.T @ null)) < 1e-10
    assert np.linalg.norm((null @ null.T) @ model.weights) < 1e-9
    assert multivariate_r2(movement, model.predict(activity)) > 1 - 1e-10

    pca = fit_pca(rng.standard_normal((100, KINEMATIC_RAW_DIM)), KINEMATIC_PCA_DIM)
    assert pca.components.shape == (KINEMATIC_RAW_DIM, KINEMATIC_PCA_DIM)
    assert 0 < pca.explained_fraction <= 1
    print("Self-test passed: rank, orthogonality, null-output, ridge, and PCA checks.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fitted balanced 12D-neural / 6D-kinematic potent-null analysis"
    )
    parser.add_argument("--animal", choices=sorted(ANIMAL_SESSIONS), default="TS")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--inner-folds", type=int, default=N_INNER_FOLDS)
    parser.add_argument(
        "--alpha-grid",
        type=parse_alpha_grid,
        default=RIDGE_GRID,
        help="comma-separated dimensionless ridge alphas",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one session with 2 folds, 1 repeat, and 2 inner folds",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.folds < 2 or args.repeats < 1 or args.inner_folds < 2:
        raise ValueError("folds>=2, repeats>=1, and inner-folds>=2 are required")

    r1_sessions, r2_sessions = ANIMAL_SESSIONS[args.animal]
    session_records = [
        (session, "R1") for session in r1_sessions
    ] + [
        (session, "R2") for session in r2_sessions
    ]
    if args.smoke_test:
        session_records = session_records[:1]
        args.folds = 2
        args.repeats = 1
        args.inner_folds = 2
        args.workers = 1
    elif args.max_sessions is not None:
        session_records = session_records[:args.max_sessions]

    missing = [
        session for session, _ in session_records
        if not (du.DATA_DIR / f"{session}_processed.nwb").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing processed NWB file(s): " + ", ".join(missing)
        )

    tasks = []
    for session_index, (session, epoch) in enumerate(session_records):
        tasks.append({
            "animal": args.animal,
            "epoch": epoch,
            "session": session,
            "session_index": session_index,
            "folds": args.folds,
            "repeats": args.repeats,
            "inner_folds": args.inner_folds,
            "alpha_grid": args.alpha_grid,
            "seed": args.seed,
        })

    workers = max(1, min(args.workers, len(tasks)))
    expected_splits = len(tasks) * args.repeats * args.folds
    print(
        f"Balanced potent/null: animal={args.animal}, sessions={len(tasks)}, "
        f"outer splits={expected_splits}, workers={workers}"
    )
    print(
        f"Design: neural {NEURAL_DIM}D; kinematics {KINEMATIC_RAW_DIM}D -> "
        f"{KINEMATIC_PCA_DIM}D; expected potent/null = 6D/6D"
    )
    started = time.perf_counter()
    completed = []
    if workers == 1:
        for task in tasks:
            result = analyze_session(task)
            completed.append(result)
            print(
                f"[{len(completed):02d}/{len(tasks):02d}] {result['epoch']} "
                f"{result['session'][4:12]}: {result['n_trials']} trials, "
                f"{result['n_units']} units, {result['elapsed_seconds']:.1f}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(analyze_session, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                print(
                    f"[{len(completed):02d}/{len(tasks):02d}] {result['epoch']} "
                    f"{result['session'][4:12]}: {result['n_trials']} trials, "
                    f"{result['n_units']} units, {result['elapsed_seconds']:.1f}s",
                    flush=True,
                )

    split_df = pd.DataFrame(
        [row for result in completed for row in result["rows"]]
    ).sort_values(["epoch", "session_date", "repeat", "fold"]).reset_index(drop=True)
    repeat_df = aggregate_rows(
        split_df, ["animal", "epoch", "session", "session_date", "repeat"]
    )
    session_df = aggregate_rows(
        split_df, ["animal", "epoch", "session", "session_date"]
    )

    epoch_records = []
    for epoch, group in session_df.groupby("epoch", sort=True):
        epoch_records.append({
            "animal": args.animal,
            "epoch": epoch,
            "n_sessions": int(len(group)),
            "mean_balanced_null_potent_ratio": float(
                group["balanced_null_potent_ratio"].mean()
            ),
            "median_balanced_null_potent_ratio": float(
                group["balanced_null_potent_ratio"].median()
            ),
            "geometric_mean_balanced_ratio": float(
                np.exp(group["log_balanced_ratio"].mean())
            ),
            "sd_log_ratio": float(group["log_balanced_ratio"].std(ddof=1))
            if len(group) > 1 else np.nan,
            "mean_readout_r2": float(group["mean_readout_r2"].mean()),
            "mean_kinematic_pca_explained": float(
                group["mean_kinematic_pca_explained"].mean()
            ),
            "mean_neural_pca_explained": float(
                group["mean_neural_pca_explained"].mean()
            ),
            "mean_potent_dimension": float(group["mean_potent_dimension"].mean()),
            "mean_null_dimension": float(group["mean_null_dimension"].mean()),
        })
    epoch_df = pd.DataFrame(epoch_records)

    permutation = exact_epoch_permutation(session_df)
    if permutation:
        permutation.update(bootstrap_epoch_difference(session_df, args.seed))
        stats_df = pd.DataFrame([permutation])
    else:
        stats_df = pd.DataFrame()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.animal.lower() + ("_smoke" if args.smoke_test else "")
    paths = {
        "splits": output_dir / f"mechanism_potent_null_balanced_splits_{suffix}.csv",
        "repeats": output_dir / f"mechanism_potent_null_balanced_repeats_{suffix}.csv",
        "sessions": output_dir / f"mechanism_potent_null_balanced_sessions_{suffix}.csv",
        "epochs": output_dir / f"mechanism_potent_null_balanced_epochs_{suffix}.csv",
        "stats": output_dir / f"mechanism_potent_null_balanced_stats_{suffix}.csv",
        "figure": output_dir / f"mechanism_potent_null_balanced_{suffix}.png",
        "config": output_dir / f"mechanism_potent_null_balanced_config_{suffix}.json",
    }
    split_df.to_csv(paths["splits"], index=False)
    repeat_df.to_csv(paths["repeats"], index=False)
    session_df.to_csv(paths["sessions"], index=False)
    epoch_df.to_csv(paths["epochs"], index=False)
    stats_df.to_csv(paths["stats"], index=False)
    if set(session_df["epoch"]) == {"R1", "R2"}:
        make_summary_figure(session_df, paths["figure"])

    elapsed = time.perf_counter() - started
    config = {
        "analysis": "cross-fitted balanced output-potent/output-null energy",
        "animal": args.animal,
        "target": TARGET,
        "bin_ms": BIN_MS,
        "neural_dimension": NEURAL_DIM,
        "kinematic_raw_dimension": KINEMATIC_RAW_DIM,
        "kinematic_pca_dimension": KINEMATIC_PCA_DIM,
        "folds": args.folds,
        "repeats": args.repeats,
        "inner_folds": args.inner_folds,
        "ridge_alpha_grid": list(args.alpha_grid),
        "seed": args.seed,
        "workers": workers,
        "sessions": [session for session, _ in session_records],
        "elapsed_seconds": elapsed,
        "interpretation": (
            "Balanced held-out null/potent activity energy, not Kaufman's "
            "prep/move-normalized tuning ratio."
        ),
    }
    paths["config"].write_text(json.dumps(config, indent=2) + "\n")

    print("\nEpoch summary")
    print(epoch_df.to_string(index=False))
    if not stats_df.empty:
        print("\nR2-R1 session-level test")
        print(stats_df.to_string(index=False))
    print("\nMaximum decomposition errors")
    for column in (
        "max_potent_null_dot",
        "projector_completeness_error",
        "activity_reconstruction_relative_error",
        "null_fixed_output_relative",
    ):
        print(f"  {column}: {split_df[column].max():.3e}")
    print(f"\nCompleted {len(split_df)} held-out splits in {elapsed:.1f}s")
    for name, path in paths.items():
        if path.exists():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
