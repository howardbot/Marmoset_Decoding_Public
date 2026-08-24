"""Master dimensionality sweep (consolidates all PCA/CCA exploration scripts).

One nested sweep, three swept factors:
  trial_mode : how CCA alignment is fit
       'average'      -> CCA on the ~20-point trial-averaged trajectory (Gallego
                         style; overfits at high K on few samples)
       'single_trial' -> CCA on phase-matched concatenated single trials
                         (hundreds of samples; robust at high K)
  K_PCS      : PCA dimensions per day, swept up to the dim that reaches ~80%
               neural variance (so the cap is data-driven).
  d          : CCA dimensions kept for decoding, 1..K_PCS.

For every (trial_mode, K_PCS, d) it cross-day decodes (Kalman, velocity) the
R1->R2 (forward / interference), R2->R1 (reverse) and R1->R1 (baseline) pairs,
and records the INTERFERENCE ASYMMETRY = (R2->R1) - (R1->R2). Also stores the
neural variance captured at each K_PCS.

This subsumes: pca_robustness_check, pca_dim_decoding, cca_singletrial_check,
cross_day_decode_vs_pca, cca_double_sweep, plot_asymmetry_persistence.

Config: locked (bin=30, butter_o2, sigma=50ms, 0828 trial-41 excluded).
Output: Results/manifold_geometry/dimension_sweep_long.csv (+ figure).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

import decoder_utils as du
from manifold_align import pca_neural, trial_average_pc, cca_align, apply_alignment
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS, EXCLUDE_TRIALS, N_PHASE_BINS,
    SMOOTH_SIGMA_MS, UNIT_QUALITIES, TRIAL_RESULTS, filter_trials,
    kalman_fit_predict, m2_per_trial,
)

warnings.filterwarnings("ignore")

BIN_SIZE_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
TARGETS = ["relative_velocity", "relative_position"]   # both, so §1 shows the position sweep too
K_LIST = [2, 5, 10, 12, 15, 20, 30, 45]  # 12 = v2 locked K_PCS; up to ~80% variance
D_LIST = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30, 45]
P_PHASE = 20                              # phases per trial for single_trial CCA
TRIAL_MODES = ["average", "single_trial"]
SEED = 0
N_R1R1_SAMPLE = 42                        # R1->R1 baseline pairs, balanced to the
                                          # R1->R2 pair count (14x3=42) for a stable
                                          # baseline without slowing the nested sweep
                                          # (running all 182 across the full sweep is ~5x).

REPO_ROOT = _THIS.parents[1]
OUT_CSV = REPO_ROOT / "Results" / "manifold_geometry" / "dimension_sweep_long.csv"
FIG_DIR = REPO_ROOT / "Results" / "manifold_geometry" / "figures"


def load_session(session, target, exclude=()):
    # Make 30ms into seconds
    bin_s = BIN_SIZE_MS / 1000.0
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_s
    io, nwb, reach = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb, reach, target, bin_size=bin_s, unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS, trial_window="start_to_peak", **SMOOTHER_KW)
    finally:
        io.close()
    X, Y, meta = filter_trials(X, Y, meta, exclude)
    Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=SMOOTH_SIGMA_MS / BIN_SIZE_MS)
    return X, Y_sm, meta

# Calculating how much cumulative variance we have captured
def cumvar(Y_sm, K):
    Yc = Y_sm - Y_sm.mean(0, keepdims=True)
    s = np.linalg.svd(Yc, compute_uv=False) ** 2
    return float(s[:K].sum() / s.sum())

# Resample all the trials in a session then return a trial trajectory list
def trial_list(Y_pc, meta, P=P_PHASE):
    out = []
    for _, idx in meta.groupby("trial_number").indices.items():
        idx = np.asarray(idx)
        if len(idx) < 3:
            continue
        ts = np.linspace(0, 1, len(idx)); tt = np.linspace(0, 1, P)
        out.append(np.column_stack([np.interp(tt, ts, Y_pc[idx, d]) for d in range(Y_pc.shape[1])]))
    return out

#
def align_full(mode, K, ca, cb, rng):
    """Return (Ya_canon, Yb_canon) full-data latents aligned A->canon, B->canon."""
    Xa, Ya, ma_, tja = ca
    Xb, Yb, mb_, tjb = cb
    if mode == "average":
        # averaged trajectory has only N_PHASE_BINS samples -> CCA n_components
        # cannot exceed that. At high K_PCS this is infeasible (the whole point:
        # averaged-trajectory CCA runs out of samples). Skip such cells.
        if K > tja.shape[0] - 1:
            return None, None
        try:
            Wa, Wb, mua, mub = cca_align(tja, tjb)
        except Exception:
            return None, None
        return apply_alignment(Ya, Wa, mua), apply_alignment(Yb, Wb, mub)
    # single_trial: phase-matched concatenation
    ta = trial_list(Ya, ma_); tb = trial_list(Yb, mb_)
    # Taking same amount of trials
    T = min(len(ta), len(tb))
    if T < 6:
        return None, None
    # Random Picking
    ia = rng.permutation(len(ta))[:T]; ib = rng.permutation(len(tb))[:T]
    # Combining
    A = np.vstack([ta[i] for i in ia]); B = np.vstack([tb[j] for j in ib])
    cca = CCA(n_components=K, scale=False, max_iter=1000)
    cca.fit(A, B)
    # project full data through fitted rotations (sklearn uses private _x_mean/_x_std)
    Ya_c = ((Ya - cca._x_mean) / cca._x_std) @ cca.x_rotations_
    Yb_c = ((Yb - cca._y_mean) / cca._y_std) @ cca.y_rotations_
    return Ya_c, Yb_c

#
def make_figure(df, out_dir, target, animal_suffix=""):
    """Two panels: (A) decode vs K_PCS at d=2 (asymmetry + mode robustness);
    (B) decode vs CCA d at K_PCS=12.

    R1->R1 and R2->R2 (within-epoch cross-day) are drawn on BOTH panels as
    descriptive references for the cross-EPOCH numbers. They are not same-session
    ceilings.
    """
    COL = {"R1->R2": "#e74c3c", "R2->R1": "#3498db", "R1->R1": "#7f8c8d", "R2->R2": "#2ca02c"}
    LS = {"average": "--", "single_trial": "-"}
    # Within-epoch cross-day references drawn behind; cross-epoch arms on top.
    CATS = ["R1->R1", "R2->R2", "R1->R2", "R2->R1"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    # A: decode vs K_PCS at d=2
    for mode in TRIAL_MODES:
        for cat in CATS:
            g = df[(df.trial_mode == mode) & (df.d == 2) & (df.pair_cat == cat)].sort_values("K_PCS")
            if g.empty:
                continue
            zo = 1 if cat == "R1->R1" else 3
            axA.plot(g.K_PCS, g.decode, LS[mode] + "o", color=COL[cat], lw=2, ms=4,
                     zorder=zo, label=f"{cat} · {mode}")
    axA.set_xlabel("PCA dim K_PCS"); axA.set_ylabel("cross-day decode corr (d=2)")
    axA.set_title("(A) Asymmetry & K_PCS robustness\nsolid=single_trial, dashed=average; "
                  "grey = R1->R1 within-epoch cross-day reference", fontsize=11)
    axA.legend(fontsize=8); axA.grid(alpha=0.3)
    # B: decode vs d at K_PCS=12
    for mode in TRIAL_MODES:
        for cat in CATS:
            g = df[(df.trial_mode == mode) & (df.K_PCS == 12) & (df.pair_cat == cat)].sort_values("d")
            if g.empty:
                continue
            zo = 1 if cat == "R1->R1" else 3
            axB.plot(g.d, g.decode, LS[mode] + "o", color=COL[cat], lw=2, ms=4,
                     zorder=zo, label=f"{cat} · {mode}")
    axB.set_xlabel("CCA dim d (K_PCS=12)"); axB.set_ylabel("cross-day decode corr")
    axB.set_title("(B) decode vs CCA dim\nwithin-epoch references shown for context; "
                  "high-d differences include weakly aligned components", fontsize=11)
    axB.legend(fontsize=8); axB.grid(alpha=0.3)
    tgt = target.replace("relative_", "")
    fig.suptitle(f"Dimension sweep ({tgt}): PCA dim x CCA dim x alignment mode — anchored on the "
                 "within-epoch cross-day references (R1->R1 grey, R2->R2 green)\n"
                 "R1->R2 (red) and R2->R1 (blue) are descriptive cross-epoch directions",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    # keep velocity at the original filename (report links reference it); position gets a suffix
    suffix = "" if target == "relative_velocity" else f"_{tgt}"
    suffix += animal_suffix
    out = out_dir / f"fig_dimension_sweep{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal", choices=sorted(ANIMAL_SESSIONS), default="TS"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGETS,
        default=TARGETS,
    )
    parser.add_argument(
        "--k-list", nargs="+", type=int, default=K_LIST
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sessions_r1, sessions_r2 = (
        list(sessions) for sessions in ANIMAL_SESSIONS[args.animal]
    )
    all_sessions = sessions_r1 + sessions_r2
    animal_suffix = "" if args.animal == "TS" else f"_{args.animal.lower()}"
    output_csv = OUT_CSV.with_name(
        f"{OUT_CSV.stem}{animal_suffix}.csv"
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # session-pair lists are target-independent; sample the R1->R1 baseline once so both
    # targets use the same baseline pairs.
    fwd = [(a, b) for a, b in product(sessions_r1, sessions_r2)]
    rev = [(b, a) for a, b in product(sessions_r1, sessions_r2)]
    r1r1_all = [
        (a, b) for a, b in product(sessions_r1, sessions_r1) if a != b
    ]
    r1r1 = [r1r1_all[i] for i in rng.permutation(len(r1r1_all))[:N_R1R1_SAMPLE]]
    r2r2 = [
        (a, b) for a, b in product(sessions_r2, sessions_r2) if a != b
    ]
    catpairs = {"R1->R2": fwd, "R2->R1": rev, "R1->R1": r1r1, "R2->R2": r2r2}

    rows = []
    for target in args.targets:
        print(f"loading + smoothing {len(all_sessions)} sessions for {target} ...")
        raw = {
            s: load_session(s, target, EXCLUDE_TRIALS.get(s, []))
            for s in all_sessions
        }
        for mode in TRIAL_MODES:
            for K in args.k_list:
                cache = {}
                for s in all_sessions:
                    X, Y_sm, meta = raw[s]
                    Ypc = pca_neural(Y_sm, k=K)[0]
                    cache[s] = (X, Ypc, meta, trial_average_pc(Ypc, meta, n_phase_bins=N_PHASE_BINS))
                var_K = float(
                    np.mean([cumvar(raw[s][1], K) for s in sessions_r1])
                )
                ds = [d for d in D_LIST if d <= K]
                for cat, pairs in catpairs.items():
                    acc = {d: [] for d in ds}
                    for a, b in pairs:
                        Ya_c, Yb_c = align_full(mode, K, cache[a], cache[b], rng)
                        if Ya_c is None:
                            continue
                        Xa, mb = cache[a][0], cache[b][2]
                        Xb = cache[b][0]
                        for d in ds:
                            try:
                                Xtec, pred = kalman_fit_predict(Xa, Ya_c[:, :d], Xb, Yb_c[:, :d], mb)
                                acc[d].append(m2_per_trial(Xtec, pred, mb))
                            except Exception:
                                pass
                    for d in ds:
                        if acc[d]:
                            rows.append({"animal": args.animal,
                                         "target": target, "trial_mode": mode, "K_PCS": K,
                                         "var_pct": var_K, "d": d, "pair_cat": cat,
                                         "decode": float(np.nanmean(acc[d])),
                                         "n_pairs": len(acc[d])})
                print(f"  [{target}|{mode}] K_PCS={K} (var={var_K:.0%}) done")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nsaved {output_csv}  ({len(df)} rows)")
    for target in args.targets:
        make_figure(
            df[df.target == target],
            FIG_DIR,
            target,
            animal_suffix,
        )

    # asymmetry table (per target)
    piv = df.pivot_table(index=["target", "trial_mode", "K_PCS", "var_pct", "d"],
                         columns="pair_cat", values="decode").reset_index()
    if {"R1->R2", "R2->R1"}.issubset(piv.columns):
        piv["asymmetry"] = piv["R2->R1"] - piv["R1->R2"]
        piv.to_csv(
            output_csv.with_name(
                f"dimension_sweep_asymmetry{animal_suffix}.csv"
            ),
            index=False,
        )
        print("\n=== ASYMMETRY (R2->R1 - R1->R2) at d=2 ===")
        sub = piv[piv.d == 2]
        for target in args.targets:
            for mode in TRIAL_MODES:
                g = sub[(sub.target == target) & (sub.trial_mode == mode)]
                if g.empty:
                    continue
                print(f"  [{target}|{mode}]")
                for _, r in g.iterrows():
                    print(f"    K={int(r.K_PCS):2d} (var {r.var_pct:.0%}): "
                          f"R1->R2={r['R1->R2']:.3f} R2->R1={r['R2->R1']:.3f} asym={r.asymmetry:+.3f}")


if __name__ == "__main__":
    main()
