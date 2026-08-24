"""Plot conditional null distributions for R1-private specificity."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[2]
IN_DIR = REPO / "Results" / "workflows" / "manifold_geometry" / "private_readout_specificity"
IN_CSV = IN_DIR / "private_specificity_null_distribution.csv"
FIG = REPO / "Results" / "workflows" / "manifold_geometry" / "figures" / "fig_private_specificity_null.png"

COLORS = {
    "r1_potent_principal": "#537895",
    "r1_output_null": "#9b9b9b",
}
LABELS = {
    "r1_potent_principal": "same-rank R1-potent principal directions",
    "r1_output_null": "same-rank R1 output-null directions",
}


def main():
    frame = pd.read_csv(IN_CSV)
    frame = frame[frame.scope == "all_r2_days"]
    observed = float(frame.observed_private.iloc[0])
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for family in ("r1_output_null", "r1_potent_principal"):
        values = frame.loc[frame.null_family == family, "statistic"]
        ax.hist(
            values, bins=80, density=True, alpha=0.48,
            color=COLORS[family], label=LABELS[family],
        )
    ax.axvline(
        observed, color="#d33f2f", linewidth=3,
        label=f"observed R1-private = {observed:+.3f}",
    )
    ax.set_xlabel("day-balanced selective forward rescue")
    ax.set_ylabel("conditional null density")
    ax.set_title(
        "R1-private rescue exceeds same-rank potent and output-null direction nulls"
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180, bbox_inches="tight")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()
