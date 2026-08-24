"""Random-subspace null for the L3 read-out principal angles.

Question: is the observed cross-day read-out rotation (~54 deg) and within-day PC-vs-readout angle
BELOW chance (=> genuine geometric preservation across days) or ~chance (=> a 2D-in-12D angle is just
large by construction and the measure is uninformative)?

Pure geometry — no data reload. Monte-Carlo random orthonormal subspaces in R^K (K=12), matched to the
observed subspace dims: cross-day = read-out(n_out) vs read-out(n_out); within-day = top-M PC vs
read-out(n_out). Compares the observed means (from the L3 CSVs) against the null distribution.

Output: Results/workflows/manifold_geometry/readout_geometry_null.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
REPO = _THIS.parents[2]
DIR = REPO / "Results" / "workflows" / "manifold_geometry"
K, M_TOP, N = 12, 3, 20000
rng = np.random.default_rng(0)


def rand_Q(n, p):
    Q, _ = np.linalg.qr(rng.standard_normal((n, p)))
    return Q[:, :p]


def mean_pa(QA, QB):
    sv = np.linalg.svd(QA.T @ QB, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(sv, -1, 1))).mean())


def null_dist(p, q, n=N):
    return np.array([mean_pa(rand_Q(K, p), rand_Q(K, q)) for _ in range(n)])


def pct(x, dist):
    return float((dist < x).mean() * 100)  # percentile of x within the null


def main():
    w = pd.read_csv(DIR / "readout_geometry_angles.csv")
    x = pd.read_csv(DIR / "readout_geometry_crossday.csv")
    n_out = int(w.n_out.iloc[0])

    null_cross = null_dist(n_out, n_out)      # read-out vs read-out (cross-day)
    null_within = null_dist(M_TOP, n_out)     # top-M PC vs read-out (within-day)
    lo_c, hi_c = np.percentile(null_cross, [2.5, 97.5])
    lo_w, hi_w = np.percentile(null_within, [2.5, 97.5])
    print(f"NULL cross-day (2x{n_out}D-in-{K}D read-out vs read-out): "
          f"mean {null_cross.mean():.1f} deg  95% [{lo_c:.1f}, {hi_c:.1f}]")
    print(f"NULL within-day (top-{M_TOP} PC vs {n_out}D read-out):    "
          f"mean {null_within.mean():.1f} deg  95% [{lo_w:.1f}, {hi_w:.1f}]")

    rows = []
    print("\n--- cross-day observed vs null (lower percentile = MORE preserved than chance) ---")
    for tgt in x.target.unique():
        for cat in ["R1->R1", "R1->R2", "R2->R1"]:
            g = x[(x.target == tgt) & (x.cat == cat)]
            if not len(g):
                continue
            m = g.mean_angle_deg.mean()
            p = pct(m, null_cross)
            print(f"  [{tgt:18s} {cat}] obs {m:.1f} deg  -> null pct {p:.0f}%  "
                  f"({'below chance (preserved)' if p < 5 else 'not below chance'})")
            rows.append(dict(kind="cross_day", target=tgt, group=cat, obs_deg=m,
                             null_mean=null_cross.mean(), null_lo=lo_c, null_hi=hi_c, null_pct=p))
    print("\n--- within-day observed vs null ---")
    for tgt in w.target.unique():
        for ep in ["R1", "R2"]:
            g = w[(w.target == tgt) & (w.epoch == ep)]
            m = g.mean_angle_deg.mean()
            p = pct(m, null_within)
            print(f"  [{tgt:18s} {ep}] obs {m:.1f} deg  -> null pct {p:.0f}%  "
                  f"({'below chance (aligned)' if p < 5 else 'not below chance'})")
            rows.append(dict(kind="within_day", target=tgt, group=ep, obs_deg=m,
                             null_mean=null_within.mean(), null_lo=lo_w, null_hi=hi_w, null_pct=p))

    pd.DataFrame(rows).to_csv(DIR / "readout_geometry_null.csv", index=False)
    print(f"\nsaved {DIR / 'readout_geometry_null.csv'}")


if __name__ == "__main__":
    main()
