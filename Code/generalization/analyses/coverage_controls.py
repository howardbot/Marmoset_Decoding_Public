"""Cause #1 (behavioural coverage / extrapolation): is R1->R2 bad simply because
R2 visits movement / neural states that R1 never sampled?

Two controls, per target, single-trial CCA (K_PCS=12), linear read-out on all D=12
canonical dims (asymmetry is decoder-invariant; linear gives exact bin<->error
alignment).

(A) ERROR vs MAHALANOBIS-to-R1 map, WITH a within-R1 baseline.
    Train read-out on epoch-A, test on epoch-B; per test bin: error ||pred-true||,
    D_kin = Mahalanobis of its kinematic state to A's kinematic distribution,
    D_neu = Mahalanobis of its canonical-neural state to A's neural distribution.
    corr(error, D_kin) is partly MECHANICAL (linear decoders err more in the tails),
    so we compare R1->R2 against the WITHIN-R1 baseline (train R1 day, test another
    R1 day). Excess = corr(R1->R2) - corr(within-R1). Excess ~ 0 => purely mechanical
    (no R2-specific extrapolation); Excess > 0 => genuine OOD penalty on R2 states.

(B) TWO-SIDED IN-SUPPORT TEST EVALUATION.
    R1->R2 is evaluated only on R2 bins inside R1's 95% training support; R2->R1
    only on R1 bins inside R2's 95% training support. Four support definitions
    are compared with the full test set:
      1. kinematic state,
      2. canonical-neural state,
      3. the intersection of kinematic and neural support,
      4. target-residual canonical-neural state after each session's own linear
         target->neural component is removed.
    These are support trims, not one-to-one distribution or count matching. PCA,
    CCA, and the decoder are fit before the test-bin masks are applied.

Output: printed summary + CSV + figure. NOTE: n(R2)=3.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

import decoder_utils as du
from manifold_align import pca_neural
from dimension_sweep import align_full
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS, EXCLUDE_TRIALS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
)

warnings.filterwarnings("ignore")
BIN_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
K, D, SEED = 12, 12, 0         # v2 re-anchor: K_PCS 15->12, decode ALL canonical dims (was top-2)
SUPPORT_PERCENTILE = 95
TARGETS = ["relative_velocity", "relative_position"]
OUT = _THIS.parents[2] / "Results" / "manifold_geometry"
FIG = OUT / "figures"
PAIR_CSV = OUT / "coverage_controls.csv"
SUMMARY_CSV = OUT / "coverage_controls_mask_summary.csv"
FIG_PATH = FIG / "fig_coverage_controls.png"

MASKS = {
    "full": "Full test set",
    "kinematic": "Kinematic support",
    "neural": "Neural support",
    "joint": "Kinematic + neural",
    "residual_neural": "Target-residual neural",
}


def load(session, target, exclude=()):
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = BIN_MS / 1000.0
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, target, bin_size=BIN_MS / 1000.0, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    Ysm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_MS)
    return X, pca_neural(Ysm, k=K)[0], meta, None


def maha(P, mu, SI):
    d = P - mu
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, SI, d), 0, None))


def support_to_train(train, test, ridge, percentile=SUPPORT_PERCENTILE):
    """Mahalanobis distances and an empirical train-support mask for test rows."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    mu = train.mean(axis=0)
    cov = np.atleast_2d(np.cov(train, rowvar=False))
    inverse_cov = np.linalg.inv(cov + ridge * np.eye(train.shape[1]))
    train_distance = maha(train, mu, inverse_cov)
    test_distance = maha(test, mu, inverse_cov)
    threshold = float(np.percentile(train_distance, percentile))
    return {
        "train_distance": train_distance,
        "test_distance": test_distance,
        "threshold": threshold,
        "mask": test_distance <= threshold,
    }


def target_residual(activity, target):
    """Remove each session's own linear target-related neural component.

    The fit is intentionally session-specific: the resulting support metric asks
    whether movement-unexplained neural states differ in coverage, rather than
    treating a changed neural-to-target map itself as residual variability.
    """
    activity = np.asarray(activity, dtype=float)
    target = np.asarray(target, dtype=float)
    design = np.column_stack([target, np.ones(len(target))])
    weights, *_ = np.linalg.lstsq(design, activity, rcond=None)
    return activity - design @ weights


def ridge_fit(Y, X, l2=1e-2):
    A = np.hstack([Y, np.ones((len(Y), 1))])
    return np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ X)


def ridge_pred(Y, W):
    return np.hstack([Y, np.ones((len(Y), 1))]) @ W


def corr_avg(t, p):
    if len(t) < 10:
        return np.nan
    return float(np.mean([np.corrcoef(t[:, j], p[:, j])[0, 1] for j in range(t.shape[1])]))


def zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0


def decode_pair(ca, cb, rng):
    """Train on A, test on B and evaluate several train-support test masks."""
    Ya, Yb = align_full("single_trial", K, ca, cb, rng)
    if Ya is None:
        return None
    Xa, Xb = ca[0], cb[0]
    pred = ridge_pred(Yb[:, :D], ridge_fit(Ya[:, :D], Xa))
    err = np.linalg.norm(pred - Xb, axis=1)

    kinematic = support_to_train(Xa, Xb, ridge=1e-6)
    neural = support_to_train(Ya[:, :D], Yb[:, :D], ridge=1e-3)
    train_residual = target_residual(Ya[:, :D], Xa)
    test_residual = target_residual(Yb[:, :D], Xb)
    residual_neural = support_to_train(train_residual, test_residual, ridge=1e-3)
    masks = {
        "full": np.ones(len(Xb), dtype=bool),
        "kinematic": kinematic["mask"],
        "neural": neural["mask"],
        "joint": kinematic["mask"] & neural["mask"],
        "residual_neural": residual_neural["mask"],
    }
    correlations = {
        name: corr_avg(Xb[mask], pred[mask])
        for name, mask in masks.items()
    }
    retained = {name: float(mask.mean()) for name, mask in masks.items()}

    return {
        "err": err,
        "Dk": kinematic["test_distance"],
        "Dn": neural["test_distance"],
        "Drn": residual_neural["test_distance"],
        "masks": masks,
        "correlations": correlations,
        "retained": retained,
        "threshold_Dk": kinematic["threshold"],
        "threshold_Dn": neural["threshold"],
        "threshold_Drn": residual_neural["threshold"],
        # Backward-compatible aliases for the original kinematic-only control.
        "inmask": masks["kinematic"],
        "corr_full": correlations["full"],
        "corr_matched": correlations["kinematic"],
    }


def safe_corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else np.nan


def run_target(target, rng, sessions_r1, sessions_r2):
    cache = {
        s: load(s, target, EXCLUDE_TRIALS.get(s, []))
        for s in sessions_r1 + sessions_r2
    }
    rows, poolR1R2, poolWR1 = [], [], []
    for a, b in product(sessions_r1, sessions_r2):
        f = decode_pair(cache[a], cache[b], rng)      # R1 -> R2
        r = decode_pair(cache[b], cache[a], rng)      # R2 -> R1
        if f is None or r is None:
            continue
        row = {
            "target": target,
            "r1_session": a,
            "r2_session": b,
            # Preserve the original columns for downstream compatibility.
            "full_R1R2": f["correlations"]["full"],
            "matched_R1R2": f["correlations"]["kinematic"],
            "full_R2R1": r["correlations"]["full"],
            "matched_R2R1": r["correlations"]["kinematic"],
            "frac_R2_OOD": 1.0 - f["retained"]["kinematic"],
            "err_corr_Dkin": safe_corr(f["err"], f["Dk"]),
            "err_corr_Dneu": safe_corr(f["err"], f["Dn"]),
            "err_corr_Dresidneu": safe_corr(f["err"], f["Drn"]),
        }
        for name in MASKS:
            row[f"{name}_R1R2"] = f["correlations"][name]
            row[f"{name}_R2R1"] = r["correlations"][name]
            row[f"retained_R2_{name}"] = f["retained"][name]
            row[f"retained_R1_{name}"] = r["retained"][name]
        for distance_name in ("Dk", "Dn", "Drn"):
            row[f"threshold_R1_{distance_name}"] = f[f"threshold_{distance_name}"]
            row[f"threshold_R2_{distance_name}"] = r[f"threshold_{distance_name}"]
        rows.append(row)
        poolR1R2.append(pd.DataFrame(dict(zerr=zscore(f["err"]), pk=pd.Series(f["Dk"]).rank(pct=True))))
    # within-R1 baseline for the (A) map
    r1pairs = [(a, c) for a, c in product(sessions_r1, sessions_r1) if a != c]
    wr = []
    # Use ALL R1xR1 pairs for a stable, rng-independent within-R1 baseline
    # (was a random N_WITHIN-pair subsample that made the baseline seed-dependent).
    for a, c in r1pairs:
        g = decode_pair(cache[a], cache[c], rng)
        if g is None:
            continue
        wr.append(safe_corr(g["err"], g["Dk"]))
        poolWR1.append(pd.DataFrame(dict(zerr=zscore(g["err"]), pk=pd.Series(g["Dk"]).rank(pct=True))))
    df = pd.DataFrame(rows)
    df.attrs["within_corr_Dkin"] = float(np.nanmean(wr))
    return df, pd.concat(poolR1R2, ignore_index=True), pd.concat(poolWR1, ignore_index=True)


def summarize_masks(pair_rows):
    """Aggregate direction scores and retained fractions for every support mask."""
    rows = []
    for target, group in pair_rows.groupby("target"):
        full_forward = float(group["full_R1R2"].mean())
        full_reverse = float(group["full_R2R1"].mean())
        full_gap = full_reverse - full_forward
        for name, label in MASKS.items():
            forward = float(group[f"{name}_R1R2"].mean())
            reverse = float(group[f"{name}_R2R1"].mean())
            gap = reverse - forward
            gap_retained = gap / full_gap if abs(full_gap) > 1e-12 else np.nan
            rows.append({
                "target": target,
                "mask": name,
                "mask_label": label,
                "forward_R1R2": forward,
                "reverse_R2R1": reverse,
                "asymmetry": gap,
                "full_asymmetry": full_gap,
                "gap_retained": gap_retained,
                "gap_closed": 1.0 - gap_retained,
                "mean_retained_R2_test": float(group[f"retained_R2_{name}"].mean()),
                "mean_retained_R1_test": float(group[f"retained_R1_{name}"].mean()),
                "n_pairs": int(len(group)),
            })
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mahalanobis coverage controls for TS or TY."
    )
    parser.add_argument(
        "--animal", choices=sorted(ANIMAL_SESSIONS), default="TS",
        help="Animal/session set to analyse (default: TS).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = (
        list(x) for x in ANIMAL_SESSIONS[args.animal]
    )
    suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summ, cR1R2, cWR1 = {}, {}, {}
    for tgt in TARGETS:
        df, p1, pw = run_target(tgt, rng, sessions_r1, sessions_r2)
        summ[tgt], cR1R2[tgt], cWR1[tgt] = df, p1, pw
        m = df.mean(numeric_only=True)
        wc = df.attrs["within_corr_Dkin"]
        print(f"\n########## {tgt} ##########")
        print("  (A) error vs OOD, baseline-corrected:")
        print(f"      corr(err, D_kin):  R1->R2 = {m.err_corr_Dkin:+.3f}   within-R1 = {wc:+.3f}   "
              f"EXCESS = {m.err_corr_Dkin - wc:+.3f}")
        print(f"      corr(err, D_neu):  R1->R2 = {m.err_corr_Dneu:+.3f}   (neural OOD)")
        print(f"      R2 bins outside R1 behavioural support = {m.frac_R2_OOD*100:.1f}%")
    pair_rows = pd.concat(summ.values(), ignore_index=True)
    pair_rows.insert(0, "animal", args.animal)
    pair_csv = OUT / f"coverage_controls{suffix}.csv"
    summary_csv = OUT / f"coverage_controls_mask_summary{suffix}.csv"
    figure_path = FIG / f"fig_coverage_controls{suffix}.png"
    pair_rows.to_csv(pair_csv, index=False)
    mask_summary = summarize_masks(pair_rows)
    mask_summary.insert(0, "animal", args.animal)
    mask_summary.to_csv(summary_csv, index=False)

    print("\n=== two-sided train-support test evaluation ===")
    display_cols = [
        "target", "mask", "forward_R1R2", "reverse_R2R1", "asymmetry",
        "gap_retained", "mean_retained_R2_test", "mean_retained_R1_test",
    ]
    print(mask_summary[display_cols].round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    nb = 10; bins = np.linspace(0, 1, nb + 1); xc = (bins[:-1] + bins[1:]) / 2
    def curve(p):
        idx = np.clip(np.digitize(p.pk, bins) - 1, 0, nb - 1)
        return p.groupby(idx).zerr.mean().reindex(range(nb)), p.groupby(idx).zerr.sem().reindex(range(nb))
    mp, sp = curve(cR1R2["relative_position"]); mw, sw = curve(cWR1["relative_position"])
    ax[0].errorbar(xc, mp, sp, marker="o", lw=2, color="#e67e22", label="R1->R2 (cross-epoch)")
    ax[0].errorbar(xc, mw, sw, marker="s", lw=2, color="#888", label="within-R1 (baseline)")
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_xlabel("kinematic Mahalanobis-to-train (percentile)")
    ax[0].set_ylabel("decode error (z, within pair)")
    ax[0].set_title("(A) position: OOD error vs within-R1 baseline\n(gap at right = R2-specific extrapolation)", fontsize=10)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
    mask_order = list(MASKS)
    colors = ["#777777", "#e67e22", "#3498db", "#8e44ad", "#27ae60"]
    w = .15
    offsets = (np.arange(len(mask_order)) - (len(mask_order) - 1) / 2) * w
    for i, tgt in enumerate(TARGETS):
        target_summary = mask_summary.set_index(["target", "mask"])
        vals = [target_summary.loc[(tgt, name), "asymmetry"] for name in mask_order]
        for j, (name, value) in enumerate(zip(mask_order, vals)):
            ax[1].bar(
                i + offsets[j], value, w, color=colors[j], edgecolor="k", linewidth=.5,
                label=(MASKS[name] if i == 0 else None),
            )
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].set_xticks(range(len(TARGETS))); ax[1].set_xticklabels([t.split("_")[1] for t in TARGETS])
    ax[1].set_ylabel("asymmetry  R2->R1 − R1->R2")
    ax[1].set_title("(B) asymmetry after two-sided train-support test filtering", fontsize=10)
    ax[1].legend(fontsize=7.5, frameon=False)
    ax[1].grid(alpha=.3, axis="y")
    fig.suptitle(
        f"{args.animal} coverage controls — kinematic, neural, joint, and "
        f"target-residual support   n(R1)={len(sessions_r1)}, n(R2)={len(sessions_r2)}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {pair_csv}\nsaved {summary_csv}\nsaved {figure_path}")


if __name__ == "__main__":
    main()
