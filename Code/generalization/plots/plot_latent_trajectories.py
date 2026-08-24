"""Latent-space trajectory alignment figure (Gallego-2020 Fig 4 style).

The review asked to be "convinced the alignment is real" by *showing the dynamics in
latent space*, not just reporting a CCA number. This plots the trial-averaged neural
population trajectory (a "canonical reach") in the top CCA dimensions, for:

  Row 1  within-R1 pair (R1a vs R1b)   — positive control: same epoch, different days
  Row 2  cross R1->R2 pair, ALIGNED    — the claim: pre- vs post-interference days align
  Row 3  the SAME R1->R2 pair, BEFORE alignment (index-matched PCs) — the contrast

Columns = the top 3 canonical dimensions over normalized reach time (0=start,1=end).
Overlap of the two days' curves after CCA (rows 1-2) vs their mismatch before (row 3)
is the visual proof that the alignment recovers genuine shared dynamics.

Config: K_PCS=12, single trial-averaged canonical reach (n_phase_bins=30), the v2 locked
pipeline (bin=30, butter_o2, sigma=50ms). Reads NWB -> needs the HatLab env.
Writes Results/manifold_geometry/figures/fig_latent_trajectories.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))          # Code/generalization
sys.path.insert(0, str(_THIS.parent.parent))   # Code

from dimension_sweep import load_session, EXCLUDE_TRIALS          # reuses locked loader
from manifold_align import pca_neural, trial_average_pc, cca_align, apply_alignment

K_PCS = 12
N_PHASE = 30
N_SHOW = 3  # top canonical dims to display

# Representative sessions (full NWB stems). Kept short + fixed for reproducibility.
R1A = "TSAL20250801_0830_staticAndStaticFree"
R1B = "TSAL20250805_0830_staticAndStaticFree001"
R2  = "TSAL20250829_0830_interferenceAndInterferenceFree001"


def avg_traj(session):
    """Trial-averaged PC trajectory (N_PHASE, K_PCS) for one session, v2 pipeline."""
    _, Y_sm, meta = load_session(session, EXCLUDE_TRIALS.get(session, []))
    Y_pc = pca_neural(Y_sm, k=K_PCS)[0]
    return trial_average_pc(Y_pc, meta, n_phase_bins=N_PHASE)


def align(traj_a, traj_b):
    """CCA-align two trial-averaged trajectories -> canonical coords (sign-matched)."""
    Wa, Wb, ma, mb = cca_align(traj_a, traj_b)
    ca = apply_alignment(traj_a, Wa, ma)
    cb = apply_alignment(traj_b, Wb, mb)
    # canonical variates are positively correlated by construction; enforce for display
    for d in range(ca.shape[1]):
        if np.corrcoef(ca[:, d], cb[:, d])[0, 1] < 0:
            cb[:, d] *= -1
    return ca, cb


def cc_per_dim(ca, cb):
    return [float(np.corrcoef(ca[:, d], cb[:, d])[0, 1]) for d in range(ca.shape[1])]


def main():
    print("loading 3 sessions ...")
    tA, tB, tR2 = avg_traj(R1A), avg_traj(R1B), avg_traj(R2)

    # Row 1: within-R1 aligned; Row 2: R1->R2 aligned; Row 3: R1->R2 before alignment
    ca_11, cb_11 = align(tA, tB)
    ca_12, cb_12 = align(tA, tR2)
    cc_11, cc_12 = cc_per_dim(ca_11, cb_11), cc_per_dim(ca_12, cb_12)

    t = np.linspace(0, 1, N_PHASE)
    # Per-dim HELD-OUT CC by category (from §1 / big_sweep_cca, single-trial splits) — the
    # honest quantification. The in-sample CC on 30-point averaged trajectories is ~1.0 by
    # overfitting and is deliberately NOT labelled here (that is the Gallego-Fig-4 visual view).
    HELDOUT = {"R1R1": [0.96, 0.65, 0.49], "R1R2": [0.91, 0.67, 0.44]}
    rows = [
        ("R1↔R1 (within-epoch)\nCCA-aligned", ca_11, cb_11, "R1R1", "0801", "0805"),
        ("R1↔R2 (cross-epoch)\nCCA-aligned",  ca_12, cb_12, "R1R2", "0801", "0829"),
        ("R1↔R2, BEFORE alignment\n(index-matched PCs)", tA, tR2, None, "0801", "0829"),
    ]
    fig, ax = plt.subplots(3, N_SHOW, figsize=(13, 9), sharex=True)
    for r, (title, A, B, cat, la, lb) in enumerate(rows):
        for d in range(N_SHOW):
            a = ax[r, d]
            a.plot(t, A[:, d], "-", color="#2c7fb8", lw=2.2, label=f"day {la}")
            a.plot(t, B[:, d], "--", color="#e6550d", lw=2.2, label=f"day {lb}")
            a.axhline(0, color="k", lw=.4, alpha=.4)
            a.grid(alpha=.25)
            if r == 0:
                a.set_title(f"canonical dim {d+1}", fontsize=11)
            if cat is not None:
                a.text(.03, .93, f"held-out CC ≈ {HELDOUT[cat][d]:.2f} (§1)", transform=a.transAxes,
                       fontsize=8.5, va="top",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.7))
            if d == 0:
                a.set_ylabel(title, fontsize=9.5)
            if r == 2:
                a.set_xlabel("normalized reach time")
    ax[0, N_SHOW - 1].legend(fontsize=9, loc="upper right")
    fig.suptitle("Latent-space reach trajectories align across days (Gallego-2020 Fig 4 style) · K_PCS=12\n"
                 "within-epoch (row 1) and cross-epoch R1→R2 (row 2) overlap after CCA; the same pair does "
                 "NOT overlap before (row 3).\nCurves are trial-averaged (near-perfect in-sample overlap by "
                 "construction); the labelled held-out CC (§1) is the conservative, honest quantification.",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir = _THIS.parent.parent.parent / "Results" / "manifold_geometry" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / "fig_latent_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)
    print("within-R1 CC:", [round(x, 2) for x in cc_11])
    print("R1->R2   CC:", [round(x, 2) for x in cc_12])

    # ---- 3D state-space view (the iconic Gallego rendering) ----
    panels = [
        ("R1↔R1 (within-epoch)\nCCA-aligned", ca_11, cb_11, "0801", "0805"),
        ("R1↔R2 (cross-epoch)\nCCA-aligned",  ca_12, cb_12, "0801", "0829"),
        ("R1↔R2\nBEFORE alignment",           tA,    tR2,   "0801", "0829"),
    ]
    fig3 = plt.figure(figsize=(16, 5.5))
    for i, (title, A, B, la, lb) in enumerate(panels):
        ax3 = fig3.add_subplot(1, 3, i + 1, projection="3d")
        ax3.plot(A[:, 0], A[:, 1], A[:, 2], "-", color="#2c7fb8", lw=2.6, label=f"day {la}")
        ax3.plot(B[:, 0], B[:, 1], B[:, 2], "-", color="#e6550d", lw=2.6, label=f"day {lb}")
        # start (o) and end (s) markers to show the reach direction
        for T, c in ((A, "#2c7fb8"), (B, "#e6550d")):
            ax3.scatter(*T[0, :3], color=c, s=55, marker="o", edgecolor="k", zorder=5)
            ax3.scatter(*T[-1, :3], color=c, s=70, marker="s", edgecolor="k", zorder=5)
        ax3.set_title(title, fontsize=11)
        ax3.set_xlabel("canon 1"); ax3.set_ylabel("canon 2"); ax3.set_zlabel("canon 3")
        ax3.legend(fontsize=8, loc="upper left")
        ax3.view_init(elev=22, azim=-60)
    fig3.suptitle("Neural reach trajectory in latent state space (Gallego-2020 Fig-4 style) · K_PCS=12 · "
                  "top-3 canonical dims\n"
                  "● = reach start, ■ = reach end. Two days overlap after CCA (within- AND cross-epoch); "
                  "they diverge before alignment.", fontsize=12, y=1.02)
    fig3.tight_layout(rect=(0, 0, 1, 0.93))
    out3 = figdir / "fig_latent_trajectories_3d.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print("saved", out3)


if __name__ == "__main__":
    main()
