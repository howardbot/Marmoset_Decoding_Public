# Literature → methods map: which paper answers which question, and where we stand



---

## A. §1 — How many PCA/CCA dimensions?
| paper | method we borrow | our script | status |
|---|---|---|---|
| **Gallego 2018** (*Nat Commun*, low-D manifolds) | manifold dim by a **loose variance floor** (m=12 ≈ 60–73 %), justified by robustness — *not* a strict threshold | `dimension_sweep.py` | done — we lock K_PCS=15, robustness-checked; highlighted |
| **Gallego 2020** (*Nat Neurosci*, long-term stability) | **fixed per-region** manifold dim (M1 10 / PMd 15 / S1 8); CCA on the full m-dim manifold | `dimension_sweep.py` | done — K_PCS=15 = their PMd value; highlighted |
| **Gao & Ganguli 2017** (dimensionality theory) | **participation ratio** / effective dimensionality | `analyses/subspace_inclusion.py` | done — PR 8.8 (R2) < 9.2 (R1); highlighted |

## B. §2 — Is the CCA alignment real?
| paper | method we borrow | our script | status |
|---|---|---|---|
| **Gallego 2020** (the cross-day CCA paper) | **PCA → CCA** alignment of two days' latent dynamics; + the **TME surrogate null** (ED7): preserve neuron/target cov, scramble time cov → CCA can't align | whole pipeline; `pipeline_v1/cca_ablation.py`; `analyses/cca_dynamics_surrogate.py` | done — ablation Δ+0.096; **TME-style surrogate** (held-out CC 0.89→~0) is our ED7 analogue; highlighted |
| **Stephen 2026** | distinguishes **aligned vs unaligned "CCA score"**; unaligned score is trial-averaging-dependent | (interpretation only) | done — §2 caveat: our single-trial held-out CC is the honest version; highlighted |
| **Degenhart 2020** (*Nat Biomed Eng*, BCI stabilization via alignment of low-D spaces) | precedent for **cross-day alignment** of neural manifolds | (precedent, not re-implemented) | reference — supports the alignment frame; title marked |

## C. §4 — Mechanism class (what *kind* of change?)
| paper | method we borrow | our script | status |
|---|---|---|---|
| **Golub 2018** (*Nat Neurosci*, "Learning by neural reassociation") | the **realignment / rescaling / reassociation** three-way classification — **our H3 conclusion is this triage** | `analyses/manifold_geometry.py` | done — the three hypothesis defs marked |
| **Kaufman 2014** (*Nat Neurosci*, "Cortical activity in the null space") | **potent/null (output-null)** decomposition and equal-dimensional Frobenius-energy comparison | `analyses/mechanism_potent_null.py` | done — cross-fitted 12D neural → 6D position/velocity/acceleration; 6D potent + 6D null; R1/R2 log-ratio difference n.s. |
| **Sadtler 2014** (*Nature*, neural constraints on learning) | **within- vs outside-manifold** test (H2 subspace inclusion) | `analyses/subspace_inclusion.py` | done — H2 rejected (R2 more compact); highlighted |
| **Oby 2018** (*PNAS*, new patterns with long-term learning) | **outside-manifold variance** as evidence of *new* dimensions (realignment/rescaling) | `analyses/manifold_geometry.py` | done — no new outside-manifold variance → not realignment/rescaling; highlighted |
| **Elsayed 2016** (*Nat Commun*, prep↔movement reorganization) | orthogonal task-epoch subspaces (conceptual basis for potent/null & reassociation) | (conceptual) | reference — supports the read-out-remapping picture; highlighted |

## D. §4 — Read-out drift theory (why a code can remap while behaviour holds)
| paper | method / idea | our script | status |
|---|---|---|---|
| **Rule & O'Leary 2022** (*PNAS*, "Self-healing codes") | stable behaviour via a **continually re-mapped read-out** of a drifting code | (conceptual support for reassociation) | reference — frames H3/H4; title + drift claim marked |
| **Rule et al. 2020** (*eLife* 51121, "stable task info from an unstable population") | population drifts while a **re-mapped read-out** preserves behaviour | (conceptual) | reference; title + drift→readout claim marked |

## E. §4 H4 — Interference vs ordinary drift
| paper | method / idea | relevance | status |
|---|---|---|---|
| **Zach 2012** (*PLoS ONE*, "M1/PMd directly reflect behavioral interference") | single-neuron evidence of **behavioural interference** in motor cortex | the **precedent for our H4 interference hypothesis** | reference — motivates the interference arm; title marked |
| **Perich, Gallego & Miller 2018** (*Neuron*, "A neural population mechanism for rapid learning") | learning via an **output-null / within-manifold** mechanism (connectivity unchanged) | learning/adaptation context for read-out change | reference; title + output-null claim marked |

## F. Corroboration (independent replication of the phenomenon)
| paper | finding | relevance | status |
|---|---|---|---|
| **Bougou 2025** (*bioRxiv*, human MC/SPL action encoding) | **motor cortex generalises asymmetrically (unidirectionally)** across two task formats | independent human-cortex anchor for our R1→R2/R2→R1 asymmetry | done — §4 inline anchor; highlighted |

---

## Cited in the report but NOT in the PDF folder (acquire if we want the full set)
- **Elsayed & Cunningham 2017** (*Nat Neurosci*, "Structure in neural population recordings") — the original **TME** methods paper our `cca_dynamics_surrogate.py` is modelled on (we have the 2016 prep/move paper instead).
- **Gao & Ganguli 2015** (*Curr Opin Neurobiol*) — cited alongside Gao et al. 2017 for the dimensionality concept; we hold the 2017 dimensionality-theory paper.
- **Rule et al. 2019** (*Curr Opin Neurobiol*, "Causes and consequences of representational drift") — cited in §4 provenance; we hold Rule 2020 (eLife) + Rule & O'Leary 2022 instead.
- **Driscoll et al. 2017** (*Cell*, parietal drift) — cited in §4 for "drift accumulates with time"; PDF not held.

All four are *supporting/conceptual* citations; the method-defining papers we actually
implement against (Gallego, Kaufman, Sadtler, Golub, Gao 2017) are all present and highlighted.
