"""Build locked position-decoder transfer matrices for TS and TY.

Rows are training days, columns are test days, and each cell is the mean
per-trial position correlation. Diagonal cells use five-fold within-day CV;
off-diagonal cells use the same phase-averaged CCA alignment as the main
cross-day sweep.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


THIS = Path(__file__).resolve()
GENERALIZATION = THIS.parent
CODE = GENERALIZATION.parent
for path in (CODE, GENERALIZATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import decoder_utils as du
from manifold_align import cca_align, apply_alignment
from big_sweep_phase2_crossday import (
    ANIMAL_SESSIONS,
    EXCLUDE_TRIALS,
    K_PCS,
    SEED,
    SMOOTHERS,
    build_cache_entry,
    kalman_fit_predict,
    m2_per_trial,
)


REPO = THIS.parents[2]
OUT_DIR = REPO / "Results" / "workflows" / "generalization"
BIN_SIZE_MS = 30
TARGET = "relative_position"
SMOOTHER_LABEL = "butter_o2"
SMOOTHER_KW = dict(SMOOTHERS)[SMOOTHER_LABEL]
N_FOLDS = 5
DEFAULT_WORKERS = min(8, max(1, (os.cpu_count() or 2) // 2))

_WORKER_CACHE = None
_WORKER_LIMITER = None


def session_date(session: str) -> str:
    prefix = session.split("_", 1)[0]
    digits = "".join(character for character in prefix if character.isdigit())
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def _init_worker(cache):
    global _WORKER_CACHE, _WORKER_LIMITER
    _WORKER_CACHE = cache
    _WORKER_LIMITER = threadpool_limits(limits=1)


def _diagonal_corr(cache) -> float:
    state = cache["X"]
    activity = cache["Y_pc"]
    meta = cache["meta"]
    scores = []
    for train_mask, test_mask in du.kfold_split_by_trial(
        meta,
        n_splits=N_FOLDS,
        random_seed=SEED,
    ):
        test_meta = meta[test_mask].reset_index(drop=True)
        test_state, prediction = kalman_fit_predict(
            state[train_mask],
            activity[train_mask],
            state[test_mask],
            activity[test_mask],
            test_meta,
        )
        scores.append(m2_per_trial(test_state, prediction, test_meta))
    return float(np.nanmean(scores))


def _off_diagonal_corr(train_cache, test_cache) -> float:
    train_weights, test_weights, train_mean, test_mean = cca_align(
        train_cache["traj"],
        test_cache["traj"],
    )
    train_activity = apply_alignment(
        train_cache["Y_pc"], train_weights, train_mean
    )[:, :K_PCS]
    test_activity = apply_alignment(
        test_cache["Y_pc"], test_weights, test_mean
    )[:, :K_PCS]
    test_state, prediction = kalman_fit_predict(
        train_cache["X"],
        train_activity,
        test_cache["X"],
        test_activity,
        test_cache["meta"],
    )
    return m2_per_trial(test_state, prediction, test_cache["meta"])


def _evaluate_pair(task):
    if _WORKER_CACHE is None:
        raise RuntimeError("worker cache was not initialized")
    train_session, test_session = task
    try:
        if train_session == test_session:
            correlation = _diagonal_corr(_WORKER_CACHE[train_session])
        else:
            correlation = _off_diagonal_corr(
                _WORKER_CACHE[train_session],
                _WORKER_CACHE[test_session],
            )
        return train_session, test_session, correlation, ""
    except Exception as exc:
        return (
            train_session,
            test_session,
            np.nan,
            f"{type(exc).__name__}: {exc}"[:300],
        )


def build_animal(animal: str, workers: int) -> tuple[pd.DataFrame, list[str]]:
    r1_sessions, r2_sessions = ANIMAL_SESSIONS[animal]
    requested_sessions = list(r1_sessions) + list(r2_sessions)
    cache = {}
    unavailable = []
    for index, session in enumerate(requested_sessions, start=1):
        print(
            f"[{animal} cache {index}/{len(requested_sessions)}] {session}",
            flush=True,
        )
        try:
            cache[session] = build_cache_entry(
                session,
                BIN_SIZE_MS,
                TARGET,
                SMOOTHER_KW,
                EXCLUDE_TRIALS.get(session, ()),
            )
        except Exception as exc:
            unavailable.append(session)
            print(
                f"  unavailable: {type(exc).__name__}: {exc}",
                flush=True,
            )

    sessions = [session for session in requested_sessions if session in cache]
    if not sessions:
        raise RuntimeError(f"no usable sessions for {animal}")
    tasks = list(product(sessions, repeat=2))
    rows = []
    worker_count = min(workers, len(tasks))
    print(
        f"[{animal}] evaluating {len(tasks)} train/test cells with "
        f"{worker_count} workers",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(cache,),
    ) as executor:
        futures = [executor.submit(_evaluate_pair, task) for task in tasks]
        progress_step = max(1, len(tasks) // 10)
        for completed, future in enumerate(as_completed(futures), start=1):
            train_session, test_session, correlation, error = future.result()
            rows.append({
                "animal": animal,
                "train_session": train_session,
                "test_session": test_session,
                "train_day": session_date(train_session),
                "test_day": session_date(test_session),
                "correlation": correlation,
                "error": error,
            })
            if completed % progress_step == 0 or completed == len(tasks):
                print(
                    f"[{animal}] completed {completed}/{len(tasks)} cells",
                    flush=True,
                )

    long_data = pd.DataFrame(rows).sort_values(
        ["train_day", "test_day"], ignore_index=True
    )
    matrix = long_data.pivot(
        index="train_day", columns="test_day", values="correlation"
    )
    ordered_dates = [session_date(session) for session in sessions]
    matrix = matrix.reindex(index=ordered_dates, columns=ordered_dates)

    suffix = animal.lower()
    long_path = OUT_DIR / f"locked_position_transfer_long_{suffix}.csv"
    matrix_path = OUT_DIR / f"locked_position_transfer_matrix_{suffix}.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_data.to_csv(long_path, index=False)
    matrix.to_csv(matrix_path)
    print(f"saved {long_path}", flush=True)
    print(f"saved {matrix_path}", flush=True)
    return matrix, unavailable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--animal",
        choices=("TS", "TY", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    animals = ("TS", "TY") if args.animal == "all" else (args.animal,)
    for animal in animals:
        matrix, unavailable = build_animal(animal, args.workers)
        print(
            f"[{animal}] matrix={matrix.shape}; "
            f"finite={int(np.isfinite(matrix.to_numpy()).sum())}; "
            f"unavailable={unavailable}",
            flush=True,
        )


if __name__ == "__main__":
    freeze_support()
    main()
