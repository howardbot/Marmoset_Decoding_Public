# Core question — which of three learning hypotheses explains R1 → R2?

The whole manifold-geometry analysis exists to decide, between three competing
hypotheses (Golub et al. 2018), *how* the neural representation of the same
static reach changed after the interference period. The three differ in whether
the **neural repertoire** (the set of activity patterns the population produces)
changes, and whether the **neural → movement mapping** changes:

| Hypothesis | What changes | Neural repertoire | Neural→movement mapping |
|------------|--------------|-------------------|--------------------------|
| **Realignment** | New activity patterns emerge along new dimensions to better drive behavior | **Changes** (grows new dims) | — |
| **Rescaling** | Variance is redistributed (push harder/softer) along the *existing* dimensions | **Changes** (variance per axis) | — |
| **Reassociation** | The *same* existing patterns get re-mapped to different intended movements | **Preserved** | **Changes** |

One-line distinction: **realignment & rescaling change the repertoire (new/rescaled patterns); reassociation keeps the repertoire and only changes the mapping.**

### How our methods discriminate them?? Not sure with this part

| Signature (our quantity) | Realignment | Rescaling | Reassociation |
|--------------------------|-------------|-----------|---------------|
| Outside-manifold variance (new directions) | **high** | low | low |
| Alignment index (subspace overlap) | **low** | high | high |
| Eigenvalue spectrum / PR / dim (per-axis variance) | maybe | **changes** | unchanged |
| Procrustes / trajectory alignment (mean-reach shape) | changed | changed | **unchanged** |
| Cross-day decoding transfer (neural→movement map) | drops (new dims unreadable) | drops (weights miscalibrated) | drops (patterns remapped) |

- **Geometry methods** (alignment, outside-variance, dim, Procrustes) test whether the **repertoire** changed → separate realignment/rescaling from reassociation.
- **Decoding methods** (cross-day transfer, F2) test whether the **mapping** changed.
- **Important:** a cross-day decoding drop happens under *all three* hypotheses, so the drop alone does **not** discriminate. The unique signature of **reassociation** is a decoding drop **combined with preserved geometry** (high alignment, low outside-variance, unchanged mean-reach shape). Decoding drop + changed geometry → realignment/rescaling; decoding drop + preserved geometry → reassociation. Our data show the latter.

---

## Main table — implemented geometric methods

| Method | Input | Reference | Question it answers |
|--------|-------------------|-----------|---------------------|
| **Participation ratio** `(Σλ)² / Σλ²` | Per session (own PC space) | Gao & Ganguli 2017 | How many *effective* dimensions does this day's manifold use? (variance concentrated on a few axes vs spread out) |
| **dim_for_80% / dim_for_90%** | Per session | Gao & Ganguli 2017 | How many PCs are needed to capture 80% / 90% of the variance? (intuitive "manifold size") |
| **Alignment index** (single-trial covariance, both directions) | Session pair | **Elsayed 2016** | How much of one day's activity variance is captured by the *other* day's manifold axes? (subspace containment, normalized to each day's own best-case) |
| **Outside-manifold variance** (both directions) | Session pair | **Oby 2019** | How much of one day's variance falls *outside* the other day's manifold? (i.e. whether a "new pattern" has emerged) |
| **Alignment index** (trial-averaged trajectory covariance, both directions) | Session pair | **Elsayed 2016** | With trial-to-trial variability averaged out — using only the mean reach trajectory — how well do the two days' manifolds align? |
| **Procrustes distance** | Session pair | **Bougou 2025** | After allowing translation + rotation + isotropic scaling, how different is the *shape* of the two days' mean reach trajectories? |
| **Canonical correlations** (top CCA correlations) | Session pair | **Gallego 2020** | When the two days are CCA-aligned, how high are the top canonical-dimension correlations? (alignment-quality diagnostic) |
| **Day-gap control** (regression extrapolation) | All pairs | **Gallego 2020** (precedent); project F3 logic | Are the R1↔R2 differences a real signal, or just ordinary time drift? (regress within-R1 metric on calendar gap, extrapolate, test R1↔R2 residuals) |

All references above were verified against the source PDFs (text highlighted in
each): Elsayed 2016 (alignment index formula), Oby 2019 (inside/outside-manifold
decomposition), Gao & Ganguli 2017 (participation ratio eq. 4, which also defines
the 80%-variance dimensionality), Bougou 2025 (Procrustes), Gallego 2020 (CCA +
canonical correlations, day-gap stability in Fig. 5). Sadtler 2014 defines the
intrinsic manifold via factor analysis but does **not** contain the participation
ratio (confirmed absent); our dimensionality metrics are PCA-based (Gao & Ganguli),
and outside-variance / manifold use PCA where Oby/Sadtler use factor analysis (FA
remains a TODO).

## Literature framing — each paper's core idea and the tool it gives us

| Reference | Core framework | Tool it provides | What it asks of R1/R2 |
|-----------|----------------|------------------|------------------------|
| **Sadtler 2014** | Neural activity lives on a low-dim "intrinsic manifold"; learning *within* the manifold is easy, *outside* is hard | Dimensionality estimate + "is activity inside the manifold?" | Is R2 still inside the R1 manifold? |
| **Oby 2019** | Long-term learning can *grow new manifold dimensions* ("emergence of new patterns") | Outside-manifold variance | Did R2 grow directions that R1 lacks? |
| **Elsayed 2016** | Two behaviors' neural activity can occupy orthogonal / shared subspaces | Alignment index (bidirectional) | Do R1 and R2 subspaces overlap or sit apart? |
| **Bougou 2025** (the read paper) | Geometric alignment / asymmetry of cross-format representations | Procrustes + cross-format decoding | Did R1↔R2 trajectory shape change / is decoding unidirectionally asymmetric? |
| **Gao & Ganguli 2017** | Neural dimensionality = participation ratio of the PCA eigenspectrum | Participation ratio + dim-for-80% | How large is each day's manifold? |
| **Gallego 2020** | Latent dynamics are stable across days within a preserved manifold; CCA aligns them | CCA alignment + canonical correlations + day-gap stability | Are R1/R2 latent dynamics the same once aligned, beyond time drift? |
| **Golub 2018** | Learning changes population activity by realignment, rescaling, or reassociation | The three competing hypotheses (the **core question**) | Which one explains R1 → R2? |
