"""Re-baseline the cross-day asymmetry against R1's OWN cross-day generalization.

The §4 "asymmetric generalisation gap" is sharpest when both transfer arms are compared not to
the within-DAY ceiling but to R1->R1 *cross-day* decoding — i.e. how well an R1-trained decoder
does on a DIFFERENT R1 day it was not trained on (same epoch, no interference, same ~gap scale).
That is the fair "ordinary day-to-day generalisation" baseline.

Reading (per-pair decode corr, single-trial CCA, K_PCS=12, all canonical dims):
  - R1->R2 (forward) BELOW the R1->R1 baseline  -> R2 is harder for an R1 decoder than an ordinary
    other R1 day: R2's representation changed in a way R1's specific read-out does not capture.
  - R2->R1 (reverse) AT/ABOVE the R1->R1 baseline -> R2's decoder generalises to R1 at least as well
    as R1 generalises to itself: R2 is a generic/transferable "core".
  - Two arms deviating in OPPOSITE directions around the same baseline is incompatible with
    symmetric drift (which would push both arms down together).

Honesty on n: R2 is only 3 sessions, so pair-level bootstrap (42 ordered R1<->R2 pairs) over-counts
(pairs sharing an R2 day are correlated). We report BOTH a pair-level bootstrap CI and a
session-clustered view (mean per R2 day, range across the 3), and let the weaker of the two govern
the claim.

Input : Results/workflows/generalization/cca_sweep_long.csv  (per-pair decode corr; no NWB needed)
Output: Results/workflows/manifold_geometry/readout_generalization_baseline.csv (+ figure)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
REPO = _THIS.parents[2]
IN_CSV = REPO / "Results" / "workflows" / "generalization" / "cca_sweep_long.csv"
OUT_CSV = REPO / "Results" / "workflows" / "manifold_geometry" / "readout_generalization_baseline.csv"
FIG = REPO / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_readout_generalization_baseline.png"

N_CCA = 12          # all canonical dims at K_PCS=12 (v2 config)
N_BOOT = 10000
SEED = 0
KEY = ["train_session", "test_session"]


def canonical_pairs(sub: pd.DataFrame) -> pd.DataFrame:
    """One row per ordered (train,test) session pair: prefer the 'exclude' (0828 trial-41 removed)
    value where it exists, else fall back to 'include' (R1->R1 pairs only exist under 'include')."""
    ex = sub[sub.outlier_mode == "exclude"]
    inc = sub[sub.outlier_mode == "include"]
    have = set(map(tuple, ex[KEY].to_numpy()))
    add = inc[~inc[KEY].apply(tuple, axis=1).isin(have)]
    return pd.concat([ex, add], ignore_index=True)


def boot_mean(x: np.ndarray, rng, n=N_BOOT) -> np.ndarray:
    """Bootstrap distribution of the mean of x (resample pairs with replacement)."""
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return x[idx].mean(axis=1)


def summarize(vals: dict[str, np.ndarray], rng) -> list[dict]:
    """Point means + pair-level bootstrap CIs for the three contrasts of interest."""
    boots = {k: boot_mean(v, rng) for k, v in vals.items()}
    out = []
    contrasts = [
        ("R1->R2 vs R1->R1 (forward vs baseline)", "R1->R2", "R1->R1"),
        ("R2->R1 vs R1->R1 (reverse vs baseline)", "R2->R1", "R1->R1"),
        ("R2->R1 vs R1->R2 (the asymmetry)",       "R2->R1", "R1->R2"),
    ]
    for name, a, b in contrasts:
        d = boots[a] - boots[b]
        out.append(dict(contrast=name,
                        mean_a=float(vals[a].mean()), n_a=len(vals[a]),
                        mean_b=float(vals[b].mean()), n_b=len(vals[b]),
                        diff=float(d.mean()),
                        lo=float(np.percentile(d, 2.5)),
                        hi=float(np.percentile(d, 97.5)),
                        p_gt0=float((d > 0).mean())))
    return out


def main():
    FIG.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    raw = pd.read_csv(IN_CSV)

    all_rows, plot_data = [], {}
    for target in ["relative_velocity", "relative_position"]:
        sub = raw[(raw.metric == "decode") & (raw.target_mode == target) & (raw.n_cca == N_CCA)]
        canon = canonical_pairs(sub)
        cats = ["R1->R1", "R1->R2", "R2->R1", "R2->R2"]
        vals = {c: canon[canon.pair_category == c]["corr"].to_numpy() for c in cats}

        print(f"\n===== {target}  (decode, K_PCS=12, all {N_CCA} canonical dims) =====")
        for c in cats:
            print(f"  {c:8s} n={len(vals[c]):3d}  mean={vals[c].mean():.3f}")

        summ = summarize({k: vals[k] for k in ["R1->R1", "R1->R2", "R2->R1"]}, rng)
        for s in summ:
            s["target"] = target
            star = "*" if (s["lo"] > 0 or s["hi"] < 0) else " "
            print(f"  [{star}] {s['contrast']:42s} diff={s['diff']:+.3f}  "
                  f"95%CI[{s['lo']:+.3f},{s['hi']:+.3f}]  P(>0)={s['p_gt0']:.3f}")
            all_rows.append(s)

        # --- session-clustered honesty check: mean per R2 day (n=3) ---
        r1r2 = canon[canon.pair_category == "R1->R2"].groupby("test_session")["corr"].mean()
        r2r1 = canon[canon.pair_category == "R2->R1"].groupby("train_session")["corr"].mean()
        base = vals["R1->R1"].mean()
        print(f"  session-clustered (n_R2={r2r1.size}) vs R1->R1 baseline {base:.3f}:")
        print(f"     R1->R2 per R2-day: {[round(v,3) for v in r1r2.values]}  "
              f"(all {'<' if (r1r2.values<base).all() else 'mixed'} baseline)")
        print(f"     R2->R1 per R2-day: {[round(v,3) for v in r2r1.values]}  "
              f"(all {'>' if (r2r1.values>base).all() else 'mixed'} baseline)")

        plot_data[target] = dict(vals=vals, base=base, r1r2=r1r2, r2r1=r2r1,
                                 summ={s["contrast"].split(" (")[0]: s for s in summ})

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

    # ---- figure: arms vs the R1->R1 cross-day baseline, both targets ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    col = {"R1->R1": "#7f7f7f", "R1->R2": "#e74c3c", "R2->R1": "#3498db", "R2->R2": "#2ca02c"}
    order = ["R1->R1", "R1->R2", "R2->R1", "R2->R2"]
    labels = {"R1->R1": "R1->R1\n(cross-day baseline)", "R1->R2": "R1->R2\n(forward)",
              "R2->R1": "R2->R1\n(reverse)", "R2->R2": "R2->R2"}
    for ax, target in zip(axes, ["relative_velocity", "relative_position"]):
        pd_ = plot_data[target]; vals = pd_["vals"]; base = pd_["base"]
        ax.axhline(base, color=col["R1->R1"], lw=1.5, ls="--", zorder=1,
                   label="R1->R1 cross-day baseline")
        for i, c in enumerate(order):
            v = vals[c]
            ax.scatter(np.full(len(v), i) + rng.normal(0, 0.05, len(v)), v,
                       color=col[c], s=26, alpha=.55, edgecolors="white", lw=.4, zorder=2)
            m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.errorbar(i, m, yerr=1.96 * se, fmt="o", color=col[c], ms=9,
                        capsize=4, lw=2, zorder=4, markeredgecolor="k")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[c] for c in order], fontsize=8.5)
        ax.set_title(target.replace("relative_", ""), fontsize=11)
        ax.grid(alpha=.3, axis="y")
        # annotate the two key contrasts
        fwd = pd_["summ"]["R1->R2 vs R1->R1"]; rev = pd_["summ"]["R2->R1 vs R1->R1"]
        ax.text(1, ax.get_ylim()[0], f"{fwd['diff']:+.3f}\n[{fwd['lo']:+.2f},{fwd['hi']:+.2f}]",
                ha="center", va="bottom", fontsize=7.5, color=col["R1->R2"])
        ax.text(2, ax.get_ylim()[0], f"{rev['diff']:+.3f}\n[{rev['lo']:+.2f},{rev['hi']:+.2f}]",
                ha="center", va="bottom", fontsize=7.5, color=col["R2->R1"])
    axes[0].set_ylabel("cross-day decode corr (per ordered pair)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Both transfer arms re-baselined on R1's OWN cross-day generalisation (R1->R1)\n"
                 "forward (R1->R2) falls FAR below the baseline; reverse (R2->R1) sits AT it "
                 "-> asymmetric around baseline, not symmetric drift (same gap both ways)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nsaved {OUT_CSV}\nsaved {FIG}")


if __name__ == "__main__":
    main()
