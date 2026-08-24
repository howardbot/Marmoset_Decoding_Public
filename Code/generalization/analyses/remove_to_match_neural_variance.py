"""Remove paired trials to match movement-residual neural variance.

This is the trial-subsetting control proposed in the 2026-07-22 meeting note:

1. Compare the final three R1 (static) days with the three R2 (interference)
   days.
2. For each R1/R2 day pair, Hungarian-match trials one-to-one using their
   phase-normalized position + velocity trajectories.
3. In each day's own 12-D PCA space, regress position + velocity out of the
   neural activity.  Greedily remove *paired* trials until the R1/R2
   trial-to-trial residual-variance ratio is as close to one as possible (while
   retaining at least 20 kinematically matched trial pairs).
4. Refit PCA, single-trial CCA, and the Kalman decoder on every retained
   subset.  Compare with equally sized random subsets of the same kinematic
   trial pairs.

The neural selection uses both sessions and is therefore a target-informed,
transductive sensitivity analysis, not a deployable or confirmatory decoder.

Outputs
-------
Results/manifold_geometry/remove_to_match_neural_variance_daily.csv
Results/manifold_geometry/remove_to_match_neural_variance_pairs.csv
Results/manifold_geometry/remove_to_match_neural_variance_summary.csv
Results/manifold_geometry/figures/fig_remove_to_match_neural_variance.png
"""
from __future__ import annotations

import os
import sys
import warnings
from itertools import combinations, product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import CCA

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from big_sweep_phase2_crossday import (
    EXCLUDE_TRIALS,
    SESSIONS_R1,
    SESSIONS_R2,
    SMOOTH_SIGMA_MS,
    TRIAL_RESULTS,
    UNIT_QUALITIES,
    filter_trials,
    kalman_fit_predict,
    m2_per_trial,
)
from manifold_align import pca_neural

warnings.filterwarnings("ignore")

BIN_MS = 30
K_PCS = 12
N_PHASE_MATCH = 30
N_PHASE_CCA = 20
MIN_RETAIN_FRACTION = 0.0
MIN_RETAIN_TRIALS = 20
MATCH_TOLERANCE = 0.02
N_RANDOM_REPS = 10
SEED = 0
SMOOTHER_KW = {
    "smoother": "butter",
    "smooth_cutoff_hz": 6.0,
    "smooth_order": 2,
}

R1_SESSIONS = SESSIONS_R1[-3:]
R2_SESSIONS = SESSIONS_R2
REPO = _THIS.parents[2]
OUT_DIR = REPO / "Results" / "manifold_geometry"
OUT_DAILY = OUT_DIR / "remove_to_match_neural_variance_daily.csv"
OUT_PAIRS = OUT_DIR / "remove_to_match_neural_variance_pairs.csv"
OUT_SUMMARY = OUT_DIR / "remove_to_match_neural_variance_summary.csv"
FIG = OUT_DIR / "figures" / "fig_remove_to_match_neural_variance.png"


def session_label(session: str) -> str:
    return session.replace("TSAL", "")[:8]


def load_session(session: str, exclude=()):
    """Load position target and smoothed population activity for one day."""
    bin_s = BIN_MS / 1000.0
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_s
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        Xp, Y, meta = du.build_decoder_dataset(
            nwb,
            reach,
            "relative_position",
            bin_size=bin_s,
            unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS,
            trial_window="start_to_peak",
            **SMOOTHER_KW,
        )
    finally:
        io.close()
    Xp, Y, meta = filter_trials(Xp, Y, meta, exclude)
    Ysm = du.smooth_neural_causal(
        Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS
    )

    # Per-trial finite difference.  Dividing by dt only changes the velocity
    # units; kinematic features are standardized before trial matching.
    Xv = np.zeros_like(Xp, dtype=float)
    for idx in meta.groupby("trial_number").indices.values():
        idx = np.asarray(idx)
        if len(idx) > 1:
            Xv[idx] = np.gradient(Xp[idx], bin_s, axis=0)
    kin = np.column_stack([Xp, Xv])

    # The selection coordinate system is locked before subsetting.  Decoder
    # PCA is separately refit on each selected subset below.
    Ypc = pca_neural(Ysm, k=K_PCS)[0]
    resid = kinematic_residual(Ypc, kin)
    return {
        "Xp": Xp,
        "Ysm": Ysm,
        "meta": meta,
        "kin": kin,
        "kin_traj": resample_trial_dict(kin, meta, N_PHASE_MATCH),
        "resid_traj": resample_trial_dict(resid, meta, N_PHASE_MATCH),
    }


def kinematic_residual(Ypc: np.ndarray, kin: np.ndarray) -> np.ndarray:
    """Remove the best linear position + velocity prediction from PC activity."""
    Kc = np.asarray(kin, dtype=float) - np.mean(kin, axis=0, keepdims=True)
    Yc = np.asarray(Ypc, dtype=float) - np.mean(Ypc, axis=0, keepdims=True)
    W, *_ = np.linalg.lstsq(Kc, Yc, rcond=None)
    return Yc - Kc @ W


def resample_trial_dict(A: np.ndarray, meta, n_phase: int) -> dict:
    """Return {trial_number: phase-resampled trajectory}."""
    target_phase = np.linspace(0.0, 1.0, n_phase)
    out = {}
    for trial, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 3:
            continue
        source_phase = np.linspace(0.0, 1.0, len(idx))
        out[trial] = np.column_stack(
            [
                np.interp(target_phase, source_phase, A[idx, d])
                for d in range(A.shape[1])
            ]
        )
    return out


def trial_variance(stack: np.ndarray) -> float:
    """Mean trial-to-trial variance across phase and neural dimensions."""
    stack = np.asarray(stack, dtype=float)
    centered = stack - stack.mean(axis=0, keepdims=True)
    return float(np.mean(centered**2))


def match_trials_by_kinematics(cache1: dict, cache2: dict):
    """Hungarian one-to-one matching of standardized position+velocity paths."""
    ids1 = np.asarray(list(cache1["kin_traj"]), dtype=object)
    ids2 = np.asarray(list(cache2["kin_traj"]), dtype=object)
    F1 = np.stack([cache1["kin_traj"][t].ravel() for t in ids1])
    F2 = np.stack([cache2["kin_traj"][t].ravel() for t in ids2])
    pooled = np.vstack([F1, F2])
    mean = pooled.mean(axis=0, keepdims=True)
    std = pooled.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    Z1 = (F1 - mean) / std
    Z2 = (F2 - mean) / std
    cost = cdist(Z1, Z2, metric="euclidean") / np.sqrt(Z1.shape[1])
    i1, i2 = linear_sum_assignment(cost)
    return ids1[i1], ids2[i2], cost[i1, i2]


def select_pairs_to_match_variance(
    stack1: np.ndarray,
    stack2: np.ndarray,
    min_fraction: float = MIN_RETAIN_FRACTION,
    min_trials: int = MIN_RETAIN_TRIALS,
    tolerance: float = MATCH_TOLERANCE,
) -> np.ndarray:
    """Greedily delete paired trials to make V2/V1 close to one.

    At each step, if V2 > V1, delete the pair with the largest excess R2
    residual energy; if V1 > V2, delete the pair with the largest excess R1
    energy.  The first subset within ``tolerance`` is returned, preserving as
    many trials as possible.  If no subset reaches tolerance, return the
    retained subset with the smallest absolute log variance ratio.
    """
    stack1 = np.asarray(stack1, dtype=float)
    stack2 = np.asarray(stack2, dtype=float)
    if stack1.shape != stack2.shape or stack1.ndim != 3:
        raise ValueError("stack1 and stack2 must have the same (trial, phase, dim) shape")
    n = len(stack1)
    if n < 2:
        raise ValueError("at least two matched trials are required")
    minimum = max(min_trials, int(np.ceil(min_fraction * n)))
    minimum = min(minimum, n)
    keep = np.arange(n)
    best = keep.copy()
    best_error = np.inf

    while True:
        A, B = stack1[keep], stack2[keep]
        v1, v2 = trial_variance(A), trial_variance(B)
        ratio = (v2 + 1e-12) / (v1 + 1e-12)
        error = abs(np.log(ratio))
        if error < best_error:
            best, best_error = keep.copy(), error
        if abs(ratio - 1.0) <= tolerance or len(keep) <= minimum:
            break

        e1 = np.mean((A - A.mean(axis=0, keepdims=True)) ** 2, axis=(1, 2))
        e2 = np.mean((B - B.mean(axis=0, keepdims=True)) ** 2, axis=(1, 2))
        excess = e2 - e1
        remove_local = int(np.argmax(excess) if v2 > v1 else np.argmin(excess))
        keep = np.delete(keep, remove_local)
    return best


def mask_for_trials(meta, trials) -> np.ndarray:
    return meta["trial_number"].isin(list(trials)).to_numpy()


def subset_pca(Ysm: np.ndarray, mask: np.ndarray, k: int = K_PCS) -> np.ndarray:
    """Fit PCA only on selected rows, then project the full session."""
    Yfit = np.asarray(Ysm[mask], dtype=float)
    mean = Yfit.mean(axis=0)
    _, _, Vt = np.linalg.svd(Yfit - mean, full_matrices=False)
    if Vt.shape[0] < k:
        raise ValueError(f"only {Vt.shape[0]} PCA dimensions available; requested {k}")
    return (np.asarray(Ysm, dtype=float) - mean) @ Vt[:k].T


def trajectories_for_ids(Ypc: np.ndarray, meta, ids, n_phase: int) -> np.ndarray:
    trajectories = resample_trial_dict(Ypc, meta, n_phase)
    missing = [t for t in ids if t not in trajectories]
    if missing:
        raise ValueError(f"missing {len(missing)} requested trials after phase resampling")
    return np.stack([trajectories[t] for t in ids])


def decode_subset(cache1: dict, cache2: dict, ids1, ids2) -> dict:
    """Refit PCA/CCA/decoder and evaluate both directions on the same subsets."""
    if len(ids1) != len(ids2):
        raise ValueError("paired trial subsets must have equal length")
    m1 = mask_for_trials(cache1["meta"], ids1)
    m2 = mask_for_trials(cache2["meta"], ids2)
    Y1pc = subset_pca(cache1["Ysm"], m1)
    Y2pc = subset_pca(cache2["Ysm"], m2)

    A = trajectories_for_ids(Y1pc, cache1["meta"], ids1, N_PHASE_CCA)
    B = trajectories_for_ids(Y2pc, cache2["meta"], ids2, N_PHASE_CCA)
    cca = CCA(n_components=K_PCS, scale=False, max_iter=5000)
    cca.fit(A.reshape(-1, K_PCS), B.reshape(-1, K_PCS))
    Y1c = ((Y1pc - cca._x_mean) / cca._x_std) @ cca.x_rotations_
    Y2c = ((Y2pc - cca._y_mean) / cca._y_std) @ cca.y_rotations_

    X1, X2 = cache1["Xp"][m1], cache2["Xp"][m2]
    Z1, Z2 = Y1c[m1], Y2c[m2]
    meta1 = cache1["meta"].loc[m1].reset_index(drop=True)
    meta2 = cache2["meta"].loc[m2].reset_index(drop=True)
    X2c, pred12 = kalman_fit_predict(X1, Z1, X2, Z2, meta2)
    X1c, pred21 = kalman_fit_predict(X2, Z2, X1, Z1, meta1)
    fwd = m2_per_trial(X2c, pred12, meta2)
    rev = m2_per_trial(X1c, pred21, meta1)
    return {"forward": fwd, "reverse": rev, "gap": rev - fwd}


def exact_two_group_permutation_p(a, b, alternative="two-sided") -> float:
    """Exact label-permutation p-value for two small independent groups."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    obs = b.mean() - a.mean()
    stats = []
    all_idx = np.arange(len(pooled))
    for ai in combinations(all_idx, len(a)):
        ai = np.asarray(ai)
        keep_a = np.zeros(len(pooled), dtype=bool)
        keep_a[ai] = True
        stats.append(pooled[~keep_a].mean() - pooled[keep_a].mean())
    stats = np.asarray(stats)
    if alternative == "greater":
        return float(np.mean(stats >= obs - 1e-12))
    if alternative == "two-sided":
        return float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))
    raise ValueError("alternative must be 'greater' or 'two-sided'")


def exact_sign_flip_p(values) -> float:
    """Two-sided exact sign-flip p-value for paired/clustered differences."""
    values = np.asarray(values, dtype=float)
    obs = abs(values.mean())
    signs = np.array(list(product([-1.0, 1.0], repeat=len(values))))
    null = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null >= obs - 1e-12))


def make_figure(daily: pd.DataFrame, pairs: pd.DataFrame):
    colors = {"R1": "#e74c3c", "R2": "#3498db"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    for epoch, x in [("R1", 0), ("R2", 1)]:
        vals = daily.loc[daily.epoch == epoch, "residual_variance"].to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax.scatter(x + jitter, vals, s=62, color=colors[epoch], zorder=3)
        ax.hlines(vals.mean(), x - 0.18, x + 0.18, color="black", lw=2.5)
    ax.set_xticks([0, 1], ["last 3 R1", "3 R2"])
    ax.set_ylabel("movement-residual neural variance")
    ax.set_title("A  Daily residual variance")

    ax = axes[1]
    for _, row in pairs.iterrows():
        ax.plot(
            [0, 1],
            [row.var_ratio_all, row.var_ratio_matched],
            color="0.72",
            lw=1,
            zorder=1,
        )
    for x, col, color in [
        (0, "var_ratio_all", "#9467bd"),
        (1, "var_ratio_matched", "#2ca02c"),
    ]:
        vals = pairs[col].to_numpy()
        ax.scatter(np.full(len(vals), x), vals, s=40, color=color, zorder=2)
        ax.hlines(vals.mean(), x - 0.18, x + 0.18, color="black", lw=2.5)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xticks([0, 1], ["kinematic\nmatch", "+ neural variance\nmatch"])
    ax.set_ylabel("R2 / R1 residual variance")
    ax.set_title("B  Remove paired trials")

    ax = axes[2]
    cols = ["gap_all", "gap_random", "gap_matched"]
    labels = ["kinematic\nmatch", "random\nequal N", "neural variance\nmatch"]
    for _, row in pairs.iterrows():
        ax.plot(range(3), row[cols].to_numpy(dtype=float), color="0.78", lw=0.9)
    for x, (col, color) in enumerate(
        zip(cols, ["#9467bd", "#7f8c8d", "#2ca02c"])
    ):
        vals = pairs[col].to_numpy()
        sem = vals.std(ddof=1) / np.sqrt(len(vals))
        ax.errorbar(x, vals.mean(), yerr=sem, fmt="o", ms=8, color=color, capsize=4)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("decode asymmetry (R2→R1 − R1→R2)")
    ax.set_title("C  Decoder refit on subsets")

    fig.suptitle("Last 3 R1 × 3 R2: kinematic and neural-variance matching", fontsize=14)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading last 3 R1 + 3 R2 sessions ...", flush=True)
    sessions = R1_SESSIONS + R2_SESSIONS
    cache = {
        s: load_session(s, EXCLUDE_TRIALS.get(s, ()))
        for s in sessions
    }

    daily_rows = []
    for epoch, group in [("R1", R1_SESSIONS), ("R2", R2_SESSIONS)]:
        for s in group:
            st = np.stack(list(cache[s]["resid_traj"].values()))
            daily_rows.append(
                {
                    "epoch": epoch,
                    "session": session_label(s),
                    "n_trials": len(st),
                    "residual_variance": trial_variance(st),
                }
            )
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT_DAILY, index=False)

    r1_daily = daily.loc[daily.epoch == "R1", "residual_variance"]
    r2_daily = daily.loc[daily.epoch == "R2", "residual_variance"]
    p_daily_2s = exact_two_group_permutation_p(r1_daily, r2_daily, "two-sided")
    p_daily_gt = exact_two_group_permutation_p(r1_daily, r2_daily, "greater")
    print(
        f"Daily residual variance: R1={r1_daily.mean():.5f}, "
        f"R2={r2_daily.mean():.5f}, exact p(two-sided)={p_daily_2s:.3f}, "
        f"p(R2>R1)={p_daily_gt:.3f}",
        flush=True,
    )

    rows = []
    for pair_index, (s1, s2) in enumerate(product(R1_SESSIONS, R2_SESSIONS), 1):
        c1, c2 = cache[s1], cache[s2]
        ids1, ids2, kin_dist = match_trials_by_kinematics(c1, c2)
        R1 = np.stack([c1["resid_traj"][t] for t in ids1])
        R2 = np.stack([c2["resid_traj"][t] for t in ids2])
        keep = select_pairs_to_match_variance(R1, R2)
        ids1_keep, ids2_keep = ids1[keep], ids2[keep]

        v1_all, v2_all = trial_variance(R1), trial_variance(R2)
        v1_match, v2_match = trial_variance(R1[keep]), trial_variance(R2[keep])
        decode_all = decode_subset(c1, c2, ids1, ids2)
        decode_match = decode_subset(c1, c2, ids1_keep, ids2_keep)

        rng = np.random.default_rng(SEED + pair_index)
        random_decodes = []
        for _ in range(N_RANDOM_REPS):
            random_keep = np.sort(rng.choice(len(ids1), len(keep), replace=False))
            random_decodes.append(
                decode_subset(c1, c2, ids1[random_keep], ids2[random_keep])
            )
        random_df = pd.DataFrame(random_decodes)
        row = {
            "r1": session_label(s1),
            "r2": session_label(s2),
            "n_kinematic_matched": len(ids1),
            "n_variance_matched": len(keep),
            "retained_fraction": len(keep) / len(ids1),
            "kinematic_distance_all": float(np.mean(kin_dist)),
            "kinematic_distance_matched": float(np.mean(kin_dist[keep])),
            "r1_variance_all": v1_all,
            "r2_variance_all": v2_all,
            "var_ratio_all": v2_all / v1_all,
            "r1_variance_matched": v1_match,
            "r2_variance_matched": v2_match,
            "var_ratio_matched": v2_match / v1_match,
            "forward_all": decode_all["forward"],
            "reverse_all": decode_all["reverse"],
            "gap_all": decode_all["gap"],
            "forward_random": random_df.forward.mean(),
            "forward_random_sd": random_df.forward.std(ddof=1),
            "reverse_random": random_df.reverse.mean(),
            "reverse_random_sd": random_df.reverse.std(ddof=1),
            "gap_random": random_df.gap.mean(),
            "gap_random_sd": random_df.gap.std(ddof=1),
            "forward_matched": decode_match["forward"],
            "reverse_matched": decode_match["reverse"],
            "gap_matched": decode_match["gap"],
        }
        rows.append(row)
        print(
            f"[{pair_index}/9] {row['r1']} vs {row['r2']}: "
            f"N {len(ids1)}->{len(keep)}, variance ratio "
            f"{row['var_ratio_all']:.3f}->{row['var_ratio_matched']:.3f}, "
            f"gap random={row['gap_random']:+.3f}, matched={row['gap_matched']:+.3f}",
            flush=True,
        )

    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT_PAIRS, index=False)

    summary_rows = []
    for condition, suffix in [
        ("kinematic_matched_all", "all"),
        ("random_equal_count", "random"),
        ("neural_variance_matched", "matched"),
    ]:
        for metric in ["forward", "reverse", "gap"]:
            vals = pairs[f"{metric}_{suffix}"].to_numpy()
            summary_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "mean": vals.mean(),
                    "sd_across_pairs": vals.std(ddof=1),
                    "sem_across_pairs": vals.std(ddof=1) / np.sqrt(len(vals)),
                    "n_pairs": len(vals),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    # Three R2-day averages are used for the paired sign-flip test so the nine
    # crossed session pairs are not falsely treated as nine independent samples.
    gap_diff_by_r2 = pairs.assign(
        gap_difference=pairs.gap_matched - pairs.gap_random
    ).groupby("r2").gap_difference.mean()
    p_gap = exact_sign_flip_p(gap_diff_by_r2)

    print("\n=== Pair summary ===")
    print(
        pairs[
            [
                "n_kinematic_matched",
                "n_variance_matched",
                "retained_fraction",
                "var_ratio_all",
                "var_ratio_matched",
                "kinematic_distance_all",
                "kinematic_distance_matched",
            ]
        ].mean().round(3).to_string()
    )
    print("\n=== Decode means ===")
    print(summary.pivot(index="condition", columns="metric", values="mean").round(3))
    print(
        f"\nMatched-minus-random gap by R2 day: "
        f"mean={gap_diff_by_r2.mean():+.3f}, exact sign-flip p={p_gap:.3f}"
    )
    print(f"\nSaved {OUT_DAILY}")
    print(f"Saved {OUT_PAIRS}")
    print(f"Saved {OUT_SUMMARY}")
    make_figure(daily, pairs)
    print(f"Saved {FIG}")


if __name__ == "__main__":
    main()
