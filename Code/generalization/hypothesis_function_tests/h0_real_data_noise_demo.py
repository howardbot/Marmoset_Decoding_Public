"""Real-data H0 noise-injection visualization.

This is a diagnostic script, not a unit test. It loads one real R1/R2 pair,
aligns the sessions with the same single-trial CCA path used by H0, then injects
Gaussian noise into the first decoder dimensions so their standard deviation is
slightly above the real-data baseline.

Outputs:
  Results/workflows/manifold_geometry/hypothesis_function_tests/h0_real_data_noise_demo.png
  Results/workflows/manifold_geometry/hypothesis_function_tests/h0_real_data_noise_demo_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
REPO = THIS.parents[3]
CODE = REPO / "Code"
GEN = CODE / "generalization"
WHY = GEN / "analyses"
for path in (str(CODE), str(GEN), str(WHY)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dimension_sweep import align_full
from why.h0_snr_control import D, K, SEED, TARGETS, decode, load
from big_sweep_phase2_crossday import EXCLUDE_TRIALS

DEFAULT_R1 = "TSAL20250813_0830_staticAndStaticFree001"
DEFAULT_R2 = "TSAL20250830_0830_interferenceAndInterferenceFree001"
OUT_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "hypothesis_function_tests"


def inject_to_target_std(Y, target_stds, rng, dims=D):
    """Add Gaussian noise so each selected dim reaches `target_stds`.

    This is the controlled version for visualization: if the real-data baseline
    std is s and target is 1.10*s, added noise std is sqrt(target^2 - s^2).
    """
    Y = np.asarray(Y, dtype=float)
    out = Y.copy()
    added = np.zeros(dims, dtype=float)
    for d in range(dims):
        current = float(out[:, d].std())
        target = float(target_stds[d])
        if target > current:
            added[d] = np.sqrt(target ** 2 - current ** 2)
            out[:, d] += rng.normal(0.0, added[d], size=out.shape[0])
    return out, added


def finite_slice(Y, n=300):
    return np.arange(min(n, len(Y)))


def plot_demo(Y_train, Y_train_noisy, Y_test, summary, out_png, n_time=300):
    fig, axes = plt.subplots(D, 2, figsize=(12, 4.2 * D))
    axes = np.atleast_2d(axes)
    idx = finite_slice(Y_train, n_time)

    for d in range(D):
        row = summary[summary["dim"] == d].iloc[0]
        ax = axes[d, 0]
        ax.plot(idx, Y_train[idx, d], lw=1.2, label="train baseline", color="#34495e")
        ax.plot(idx, Y_train_noisy[idx, d], lw=1.0, alpha=0.75, label="train + noise", color="#e67e22")
        ax.plot(idx, Y_test[idx, d], lw=1.0, alpha=0.65, label="test reference", color="#3498db")
        ax.set_title(
            f"Canonical dim {d + 1}: first {len(idx)} bins\n"
            f"std {row.baseline_std:.3f} -> {row.noisy_std:.3f} "
            f"(target {row.target_std:.3f})",
            fontsize=10,
        )
        ax.set_xlabel("sample bin")
        ax.set_ylabel("aligned latent value")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[d, 1]
        bins = np.linspace(
            np.nanmin([Y_train[:, d].min(), Y_train_noisy[:, d].min(), Y_test[:, d].min()]),
            np.nanmax([Y_train[:, d].max(), Y_train_noisy[:, d].max(), Y_test[:, d].max()]),
            45,
        )
        ax.hist(Y_train[:, d], bins=bins, density=True, alpha=0.55, label="train baseline", color="#34495e")
        ax.hist(Y_train_noisy[:, d], bins=bins, density=True, alpha=0.45, label="train + noise", color="#e67e22")
        ax.hist(Y_test[:, d], bins=bins, density=True, alpha=0.35, label="test reference", color="#3498db")
        ax.set_title(f"Canonical dim {d + 1}: distribution", fontsize=10)
        ax.set_xlabel("aligned latent value")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("H0 real-data noise injection: baseline vs slightly higher std", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cache_r1 = load(args.r1_session, args.target, EXCLUDE_TRIALS.get(args.r1_session, []))
    cache_r2 = load(args.r2_session, args.target, EXCLUDE_TRIALS.get(args.r2_session, []))
    Y_r1, Y_r2 = align_full("single_trial", K, cache_r1, cache_r2, rng)
    if Y_r1 is None:
        raise RuntimeError("single-trial CCA alignment failed for the selected pair")

    target_stds = Y_r1[:, :D].std(axis=0) * args.std_multiplier
    Y_r1_noisy, added_noise_std = inject_to_target_std(Y_r1, target_stds, rng, dims=D)

    baseline_decode = decode(cache_r1[0], Y_r1, cache_r2[0], Y_r2, cache_r2[2])
    noisy_decode = decode(cache_r1[0], Y_r1_noisy, cache_r2[0], Y_r2, cache_r2[2])

    rows = []
    for d in range(D):
        rows.append({
            "target": args.target,
            "train_session": args.r1_session,
            "test_session": args.r2_session,
            "dim": d,
            "std_multiplier": args.std_multiplier,
            "baseline_std": float(Y_r1[:, d].std()),
            "target_std": float(target_stds[d]),
            "added_noise_std": float(added_noise_std[d]),
            "noisy_std": float(Y_r1_noisy[:, d].std()),
            "test_std": float(Y_r2[:, d].std()),
            "baseline_decode": float(baseline_decode),
            "noisy_decode": float(noisy_decode),
        })
    summary = pd.DataFrame(rows)

    tag = (
        f"h0_real_data_noise_demo_{args.target}_"
        f"{args.r1_session.split('_')[0][-4:]}_to_{args.r2_session.split('_')[0][-4:]}"
    )
    out_csv = OUT_DIR / f"{tag}_summary.csv"
    out_png = OUT_DIR / f"{tag}.png"
    summary.to_csv(out_csv, index=False)
    plot_demo(Y_r1, Y_r1_noisy, Y_r2, summary, out_png, n_time=args.n_time)

    print(summary.round(4).to_string(index=False))
    print(f"\nsaved {out_csv}")
    print(f"saved {out_png}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r1-session", default=DEFAULT_R1)
    p.add_argument("--r2-session", default=DEFAULT_R2)
    p.add_argument("--target", choices=TARGETS, default="relative_position")
    p.add_argument("--std-multiplier", type=float, default=1.10)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n-time", type=int, default=300)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
