"""Methods flow chart for the 'Asymmetrical generalization' slide — up to how
cross-day generalization is done (no mechanism). Two tiers:
  per-session processing  ->  cross-day generalization (align + decode, forward/reverse).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "Results" / "generalization" / "figures" / "fig_methods_flowchart.png"

NEU = "#dbe9f6"; NEU_E = "#3b78b8"     # neural stream (blue)
MOV = "#fdead2"; MOV_E = "#d98a2b"     # movement stream (orange)
GEN = "#dcefe0"; GEN_E = "#3c9f5b"     # generalization (green)

fig, ax = plt.subplots(figsize=(13.5, 6.2))
ax.set_xlim(0, 12.5); ax.set_ylim(-2.4, 4.4); ax.axis("off")

W, H = 2.35, 1.05

def box(x, y, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.14",
                                fc=fc, ec=ec, lw=2.2, zorder=2))
    ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=11.5, zorder=3)
    return (x, y)

def arrow(p_from, p_to, side_from="right", side_to="left", text=None, color="#555"):
    fx = p_from[0] + (W if side_from == "right" else W / 2 if side_from == "bottom" else 0)
    fy = p_from[1] + (H / 2 if side_from in ("right", "left") else 0)
    tx = p_to[0] + (0 if side_to == "left" else W / 2 if side_to == "top" else W)
    ty = p_to[1] + (H / 2 if side_to in ("right", "left") else H)
    ax.add_patch(FancyArrowPatch((fx, fy), (tx, ty), arrowstyle="-|>", mutation_scale=20,
                                 lw=2.2, color=color, zorder=1,
                                 connectionstyle="arc3,rad=0" if side_from != "bottom" else "arc3,rad=0"))
    if text:
        ax.text((fx + tx) / 2, (fy + ty) / 2 + 0.18, text, ha="center", va="bottom",
                fontsize=9.5, color=color, style="italic")

# --- neural stream ---
ax.text(0.1, 4.2, "NEURAL", fontsize=11.5, color=NEU_E, fontweight="bold")
a = box(0.1, 3.1, "Spikes\n(good + MUA)", NEU, NEU_E)
b = box(3.1, 3.1, "30 ms bins", NEU, NEU_E)
c = box(6.1, 3.1, "Causal Gaussian\nsmooth  σ = 50 ms", NEU, NEU_E)
d = box(9.1, 3.1, "PCA →\n12-D manifold", NEU, NEU_E)
for p, q in [(a, b), (b, c), (c, d)]:
    arrow(p, q)

# --- movement stream (the DECODE TARGET; this is where the swept 'smoother' lives) ---
ax.text(0.1, 2.7, "MOVEMENT  (decode target)", fontsize=11.5, color=MOV_E, fontweight="bold")
m1 = box(0.1, 1.6, "Reach pose\n(r-wrist)", MOV, MOV_E)
m2 = box(3.1, 1.6, "Relative position\n(vs shoulder)", MOV, MOV_E)
m3 = box(6.1, 1.6, "Butterworth 6 Hz\n(swept smoother)", MOV, MOV_E)
m4 = box(9.1, 1.6, "Position &\nvelocity", MOV, MOV_E)
for p, q in [(m1, m2), (m2, m3), (m3, m4)]:
    arrow(p, q, color=MOV_E)

# --- cross-day generalization ---
ax.text(0.1, 0.55, "CROSS-DAY GENERALIZATION", fontsize=11.5, color=GEN_E, fontweight="bold")
f = box(3.1, -0.55, "CCA align\ntwo days", GEN, GEN_E)
g = box(6.1, -0.55, "Kalman decode\ntrain day → test day", GEN, GEN_E)
h = box(9.1, -0.55, "Cross-day\ndecode corr", GEN, GEN_E)
arrow(d, f, side_from="bottom", side_to="top", color=NEU_E)
arrow(m4, g, side_from="bottom", side_to="top", color=MOV_E)
ax.text(4.28, 0.72, "neural manifolds,\ntwo days", ha="center", va="bottom", fontsize=8.5,
        color=NEU_E, style="italic")
ax.text(8.05, 0.72, "decode\ntarget", ha="center", va="bottom", fontsize=8.5,
        color=MOV_E, style="italic")
for p, q in [(f, g), (g, h)]:
    arrow(p, q, color=GEN_E)

# --- footer: design + forward/reverse ---
ax.text(6.25, -1.85,
        "Days:  R1 = 14 pre-interference   →   2-week gap   →   R2 = 3 return\n"
        "forward = train R1 → test R2      reverse = train R2 → test R1",
        ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f6f2e9", ec="#c8b98e", lw=1.5))

fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print("saved", OUT)
