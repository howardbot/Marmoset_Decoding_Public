"""D8-proper — same-session split-half CCA ceiling + normalized cross-day similarity (Sami's D section).

The held-out cross-day canonical correlation (§1/§2) is compared with an empirical same-session
split-half ceiling: align two independent halves of the SAME session (same day, only trial-sampling
noise). The normalized ratio places across-day CC on that descriptive scale per canonical dim; it is
not a test of statistical equivalence.

Trial-count matched (this matters — fewer trials -> noisier trajectory -> lower CC): every trajectory
is built from |trials|/4. Same-session: split session s into 4 quarters q0..q3; fit CCA on (traj q0, traj q1),
evaluate held-out CC on (traj q2, traj q3) — two same-day subsets. Cross-day (a,b): 4 quarters each;
fit CCA on (traj a0, traj b0), eval on (traj a1, traj b1). Same quarter size everywhere.

Uses the trajectory / held-out-CC machinery from cca_dynamics_surrogate (K_PCS=12).
Output: Results/workflows/manifold_geometry/d8_normalized_similarity.csv (+ figure).
"""
from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from cca_dynamics_surrogate import load_cache, traj, cc_dim, K, SESSIONS_R1, SESSIONS_R2

warnings.filterwarnings("ignore")
N_SPLITS = 100
SEED = 0
REPO = _THIS.parents[2]
OUT_CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "d8_normalized_similarity.csv"
FIG = REPO / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_d8_normalized_similarity.png"


def quarters(cache, rng):
    """4 disjoint equal-size trial subsets of one session (None if too few trials)."""
    t = np.array(sorted(cache["meta"]["trial_number"].unique()))
    p = rng.permutation(t)
    n = len(p) // 4
    if n < 3:
        return None
    return [p[i * n:(i + 1) * n] for i in range(4)]


def within_ceiling(cache, rng, n=N_SPLITS):
    """Held-out CC aligning two same-day subsets (q0,q1 fit; q2,q3 eval) -> (K,) mean per dim."""
    acc = []
    for _ in range(n):
        q = quarters(cache, rng)
        if q is None:
            continue
        acc.append(cc_dim(traj(cache, q[0]), traj(cache, q[1]),
                          traj(cache, q[2]), traj(cache, q[3])))
    return np.nanmean(acc, axis=0) if acc else None


def cross_cc(ca, cb, rng, n=N_SPLITS):
    """Held-out cross-day CC, quarter-sized to match the same-session ceiling -> (K,) mean per dim."""
    acc = []
    for _ in range(n):
        qa, qb = quarters(ca, rng), quarters(cb, rng)
        if qa is None or qb is None:
            continue
        acc.append(cc_dim(traj(ca, qa[0]), traj(cb, qb[0]),
                          traj(ca, qa[1]), traj(cb, qb[1])))
    return np.nanmean(acc, axis=0) if acc else None


def main():
    FIG.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    caches = {s: load_cache(s) for s in SESSIONS_R1 + SESSIONS_R2}
    print("loaded", len(caches), "sessions")

    # Same-session split-half ceilings, averaged per epoch.
    ceil = {ep: np.nanmean([within_ceiling(caches[s], rng)
                            for s in sess], axis=0)
            for ep, sess in [("R1", SESSIONS_R1), ("R2", SESSIONS_R2)]}
    # across-day (R1->R2), averaged over all pairs
    cross = np.nanmean([cross_cc(caches[a], caches[b], rng)
                        for a, b in product(SESSIONS_R1, SESSIONS_R2)], axis=0)

    dims = np.arange(1, K + 1)
    df = pd.DataFrame({
        "dim": dims,
        "ceiling_R1_withinday": ceil["R1"],
        "ceiling_R2_withinday": ceil["R2"],
        "cross_R1toR2": cross,
        "norm_vs_R1ceiling": cross / ceil["R1"],
        "norm_vs_R2ceiling": cross / ceil["R2"],
    })
    df.to_csv(OUT_CSV, index=False)

    print("\nSame-session split-half ceiling vs across-day CC (normalized similarity), K_PCS=12")
    print("dim  R1ceil  R2ceil  R1->R2   norm/R1ceil  norm/R2ceil")
    for _, r in df.iterrows():
        # normalized ratio is only meaningful where the ceiling is above the noise floor (~dim 1-4)
        tag = "" if r.ceiling_R1_withinday > 0.15 else "   (ceiling at floor -> ratio n/a)"
        print(f"{int(r.dim):>3}  {r.ceiling_R1_withinday:>6.3f}  {r.ceiling_R2_withinday:>6.3f}  "
              f"{r.cross_R1toR2:>6.3f}   {r.norm_vs_R1ceiling:>9.2f}   {r.norm_vs_R2ceiling:>9.2f}{tag}")

    # figure: ceilings + cross-day CC (left), normalized ratio for the meaningful dims (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(dims, ceil["R1"], "-o", color="#7f7f7f", label="R1 same-session ceiling")
    ax1.plot(dims, ceil["R2"], "-o", color="#2ca02c", label="R2 same-session ceiling")
    ax1.plot(dims, cross, "-o", color="#e74c3c", label="R1→R2 cross-day")
    ax1.axhline(0, color="k", lw=.5)
    ax1.set_xlabel("canonical dim"); ax1.set_ylabel("held-out canonical correlation")
    ax1.set_title("Same-session split-half ceiling vs cross-day CC (matched trial counts)", fontsize=10)
    ax1.set_xticks(dims); ax1.legend(fontsize=9); ax1.grid(alpha=.3)

    meaningful = df[df.ceiling_R1_withinday > 0.15]
    ax2.bar(meaningful.dim - 0.17, meaningful.norm_vs_R1ceiling, width=0.34,
            color="#7f7f7f", label="cross-day / R1 ceiling")
    ax2.bar(meaningful.dim + 0.17, meaningful.norm_vs_R2ceiling, width=0.34,
            color="#2ca02c", label="cross-day / R2 ceiling")
    ax2.axhline(1.0, color="k", ls="--", lw=1, label="= same-session mean")
    ax2.set_xlabel("canonical dim"); ax2.set_ylabel("normalized similarity")
    ax2.set_title("Cross-day CC as a fraction of the same-session ceiling\n"
                  "(only dims with ceiling above the noise floor)", fontsize=10)
    ax2.set_xticks(meaningful.dim.astype(int)); ax2.legend(fontsize=9); ax2.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nsaved {OUT_CSV}\nsaved {FIG}")


if __name__ == "__main__":
    main()
