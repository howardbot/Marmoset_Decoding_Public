"""CCA sweep: both decoding performance and alignment quality vs # CCA dims.

One sweep, one x-axis (number of canonical dimensions d = 1..15, PCA fixed at
15), two metrics measured for every cross-day (off-diagonal) session pair:

  metric = "decode"  -- cross-day decoding corr (Kalman, lag=0) using the first
                        d canonical dimensions. Recorded for both targets
                        (relative_velocity, relative_position). Direction
                        matters: train->test is a directed pair.
  metric = "cca"     -- held-out canonical correlation of the d-th canonical
                        dimension: fit CCA on one random half of each day's
                        trials, score the correlation on the other half, average
                        over N_REPEATS splits. Target-independent. Held-out is
                        essential because CCA maximises in-sample correlation by
                        construction. CC is direction-symmetric.

Everything else is pinned to LOCKED_CONFIG (bin=30, butter_o2, lag=0, Kalman;
0828 trial-41 excluded for 0828-involving pairs).

Output (one long CSV):
  Results/workflows/generalization/cca_sweep_long.csv
    columns: metric, target_mode, train_session, test_session, pair_category,
             outlier_mode, n_cca, corr
    - decode rows: target_mode in {relative_velocity, relative_position}
    - cca rows:    target_mode = "na"
    pair_category is the directed epoch pair (R1->R1 / R1->R2 / R2->R1 / R2->R2)
    for both metrics; the CC plot collapses direction (symmetric) into the mean.

Checkpointing: decode and cca passes each skip work already present in the CSV.
Plotting is in plot_cca_sweep.py.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS))

import decoder_utils as du
from manifold_align import pca_neural, trial_average_pc, cca_align, apply_alignment
from Neural_Decoding.decoders import KalmanFilterRegression
from big_sweep_phase2_crossday import (
    ALL_SESSIONS, EXCLUDE_TRIALS, K_PCS, N_PHASE_BINS, SMOOTH_SIGMA_MS,
    UNIT_QUALITIES, TRIAL_RESULTS, kalman_fit_predict, m2_per_trial,
    filter_trials, pair_involves_0828,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Locked config
# ---------------------------------------------------------------------------
BIN_SIZE_MS = 30
SMOOTHER_KW = {"smoother": "butter", "smooth_cutoff_hz": 6.0, "smooth_order": 2}
TARGET_MODES = ["relative_velocity", "relative_position"]
CCA_DIMS = list(range(1, K_PCS + 1))  # 1..15
N_REPEATS = 10                        # random trial-half splits for held-out CC
SEED = 0

REPO_ROOT = _THIS.parents[1]
OUT_CSV = REPO_ROOT / "Results" / "workflows" / "generalization" / "cca_sweep_long.csv"
N_WORKERS = 2  # decode pass: one worker per target


def epoch_of(session):
    # R2 = post-interference return sessions (named "...interference..."); R1 = "...static...".
    # Naming-based so new R2 sessions (e.g. 0830) are never missed (the old hard-coded
    # 0828/0829 list silently mislabelled 0830 as R1).
    return "R2" if "interference" in session.lower() else "R1"


def pair_category(train_session, test_session):
    return f"{epoch_of(train_session)}->{epoch_of(test_session)}"


# ---------------------------------------------------------------------------
# Cache (PCA fixed at K_PCS). X depends on target; Y_pc/traj are neural-only.
# ---------------------------------------------------------------------------
def build_cache(session, target_mode, exclude_trial_nums=()):
    bin_size_s = BIN_SIZE_MS / 1000.0
    du.SESSION = session
    du.PROCESSED_NWB = du.DATA_DIR / f"{session}_processed.nwb"
    du.BIN_SIZE_SECONDS = bin_size_s

    io, nwb_prc, reach_tbl = du.load_nwb_and_reach()
    try:
        X, Y, meta = du.build_decoder_dataset(
            nwb_prc, reach_tbl, target_mode,
            bin_size=bin_size_s,
            unit_qualities=UNIT_QUALITIES,
            trial_results=TRIAL_RESULTS,
            trial_window="start_to_peak",
            **SMOOTHER_KW,
        )
    finally:
        io.close()

    X, Y, meta = filter_trials(X, Y, meta, exclude_trial_nums)
    sigma_bins = SMOOTH_SIGMA_MS / BIN_SIZE_MS
    Y_sm = du.smooth_neural_causal(Y, meta, sigma_bins=sigma_bins)
    Y_pc, _, _ = pca_neural(Y_sm, k=K_PCS)
    traj = trial_average_pc(Y_pc, meta, n_phase_bins=N_PHASE_BINS)
    return {"X": X, "Y_pc": Y_pc, "meta": meta, "traj": traj}


# ---------------------------------------------------------------------------
# DECODE pass: cross-day decoding corr at each d (lag=0 Kalman)
# ---------------------------------------------------------------------------
def decode_pair_dims(tr_cache, te_cache):
    """{d: corr} for d in CCA_DIMS, Kalman in first-d canonical space."""
    W_tr, W_te, m_tr, m_te = cca_align(tr_cache["traj"], te_cache["traj"])
    Y_tr = apply_alignment(tr_cache["Y_pc"], W_tr, m_tr)
    Y_te = apply_alignment(te_cache["Y_pc"], W_te, m_te)
    X_tr, meta_tr = tr_cache["X"], tr_cache["meta"]
    X_te, meta_te = te_cache["X"], te_cache["meta"]
    if len(X_tr) < 50 or len(X_te) < 20:
        return {d: np.nan for d in CCA_DIMS}
    out = {}
    for d in CCA_DIMS:
        try:
            X_te_c, pred = kalman_fit_predict(X_tr, Y_tr[:, :d], X_te, Y_te[:, :d], meta_te)
            out[d] = m2_per_trial(X_te_c, pred, meta_te)
        except Exception:
            out[d] = np.nan
    return out


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    warnings.filterwarnings("ignore")


def run_decode_target(args):
    target_mode, pending = args  # pending: list of (tr, te, outlier_mode)
    try:
        need_exc = {s for (tr, te, om) in pending if om == "exclude"
                    for s in (tr, te) if s in EXCLUDE_TRIALS}
        cache_inc = {s: build_cache(s, target_mode, ()) for s in ALL_SESSIONS}
        cache_exc = {s: build_cache(s, target_mode, EXCLUDE_TRIALS[s]) for s in need_exc}
        rows = []
        for tr, te, om in pending:
            tc = cache_exc[tr] if (om == "exclude" and tr in cache_exc) else cache_inc[tr]
            ec = cache_exc[te] if (om == "exclude" and te in cache_exc) else cache_inc[te]
            for d, c in decode_pair_dims(tc, ec).items():
                rows.append({
                    "metric": "decode", "target_mode": target_mode,
                    "train_session": tr, "test_session": te,
                    "pair_category": pair_category(tr, te), "outlier_mode": om,
                    "n_cca": d, "corr": c,
                })
        return ("ok", target_mode, rows, None)
    except Exception as e:
        return ("fail", target_mode, None, f"{type(e).__name__}:{e}"[:200])


# ---------------------------------------------------------------------------
# CCA pass: held-out canonical correlation at each d (target-independent)
# ---------------------------------------------------------------------------
def traj_from_trials(Y_pc, meta, trials):
    mask = meta["trial_number"].isin(trials).to_numpy()
    return trial_average_pc(Y_pc[mask], meta[mask].reset_index(drop=True),
                            n_phase_bins=N_PHASE_BINS)


def heldout_cc(cache_a, cache_b, n_repeats=N_REPEATS, seed=SEED):
    rng = np.random.default_rng(seed)
    ta = np.array(sorted(cache_a["meta"]["trial_number"].unique()))
    tb = np.array(sorted(cache_b["meta"]["trial_number"].unique()))

    def halves(t):
        p = rng.permutation(t); h = len(p) // 2
        return p[:h], p[h:]

    ccs = []
    for _ in range(n_repeats):
        a_fit, a_ev = halves(ta); b_fit, b_ev = halves(tb)
        if min(len(a_fit), len(a_ev), len(b_fit), len(b_ev)) < 2:
            continue
        try:
            W_a, W_b, m_a, m_b = cca_align(
                traj_from_trials(cache_a["Y_pc"], cache_a["meta"], a_fit),
                traj_from_trials(cache_b["Y_pc"], cache_b["meta"], b_fit),
            )
            ca = (traj_from_trials(cache_a["Y_pc"], cache_a["meta"], a_ev) - m_a) @ W_a
            cb = (traj_from_trials(cache_b["Y_pc"], cache_b["meta"], b_ev) - m_b) @ W_b
            ccs.append(np.array([np.corrcoef(ca[:, d], cb[:, d])[0, 1]
                                 for d in range(ca.shape[1])]))
        except Exception:
            continue
    return np.nanmean(np.vstack(ccs), axis=0) if ccs else np.full(K_PCS, np.nan)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def load_done(csv_path):
    """Return (decode_cells, cca_cells) already present."""
    if not csv_path.exists():
        return set(), set()
    df = pd.read_csv(csv_path, usecols=lambda c: c in (
        "metric", "target_mode", "train_session", "test_session"))
    dec = df[df["metric"] == "decode"]
    cca = df[df["metric"] == "cca"]
    dec_cells = set(map(tuple, dec[["target_mode", "train_session", "test_session"]]
                        .drop_duplicates().to_numpy()))
    cca_cells = set(map(tuple, cca[["train_session", "test_session"]]
                        .drop_duplicates().to_numpy()))
    return dec_cells, cca_cells


def append_rows(csv_path, rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dec_done, cca_done = load_done(OUT_CSV)
    print(f"[resume] decode cells={len(dec_done)}  cca cells={len(cca_done)}")
    t0 = time.time()

    # ---- DECODE pass (parallel over targets) ----
    work = []
    for tm in TARGET_MODES:
        pending = []
        for tr in ALL_SESSIONS:
            for te in ALL_SESSIONS:
                if tr == te:
                    continue
                om = "exclude" if pair_involves_0828(tr, te) else "include"
                if (tm, tr, te) in dec_done:
                    continue
                pending.append((tr, te, om))
        if pending:
            work.append((tm, pending))
            print(f"[decode plan] {tm}: {len(pending)} pairs x {len(CCA_DIMS)} dims")

    if work:
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as pool:
            futs = {pool.submit(run_decode_target, w): w for w in work}
            for fut in as_completed(futs):
                status, tm, rows, err = fut.result()
                if status == "ok":
                    append_rows(OUT_CSV, rows)
                    print(f"[decode done] {tm} -> {len(rows)} rows ({(time.time()-t0)/60:.1f} min)")
                else:
                    print(f"[decode FAIL] {tm}: {err}")

    # ---- CCA pass (held-out CC; unordered pairs, target-independent) ----
    unordered = [p for p in combinations(ALL_SESSIONS, 2)
                 if (p[0], p[1]) not in cca_done]
    if unordered:
        print(f"[cca plan] {len(unordered)} unordered pairs x {len(CCA_DIMS)} dims")
        cache_inc = {s: build_cache(s, TARGET_MODES[0], ()) for s in ALL_SESSIONS}
        cache_exc = {s: build_cache(s, TARGET_MODES[0], EXCLUDE_TRIALS[s]) for s in EXCLUDE_TRIALS}
        rows = []
        for a, b in unordered:
            om = "exclude" if any("20250828" in s for s in (a, b)) else "include"
            ca = cache_exc[a] if (om == "exclude" and a in cache_exc) else cache_inc[a]
            cb = cache_exc[b] if (om == "exclude" and b in cache_exc) else cache_inc[b]
            cc = heldout_cc(ca, cb)
            for d in CCA_DIMS:
                rows.append({
                    "metric": "cca", "target_mode": "na",
                    "train_session": a, "test_session": b,
                    "pair_category": pair_category(a, b), "outlier_mode": om,
                    "n_cca": d, "corr": float(cc[d - 1]),
                })
        append_rows(OUT_CSV, rows)
        print(f"[cca done] -> {len(rows)} rows")

    if not work and not unordered:
        print("Nothing to do.")
    print(f"\nDONE. Wall clock: {(time.time()-t0)/60:.1f} min\nCSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
