"""Manifold alignment helpers for cross-day decoder analysis.

Three pieces:
  1. PCA per session (raw spike-count matrix -> k-dim neural manifold)
  2. Trial-averaged PC trajectory (resample each reach to a common phase grid,
     then average across reaches -> a single average reach trajectory in PC space)
  3. CCA alignment of two days' trial-averaged trajectories, yielding rotation
     matrices that map each day's PC space into a shared canonical space
     (Gallego et al. 2020, Nat Neurosci 23:260-270).

Convention:
  - PCA centers the data, so the returned mean is the per-unit mean to subtract.
  - CCA is fit on the trial-averaged trajectories (which are NOT zero-mean);
    we store the per-day trajectory mean and apply it when projecting the full
    PC time series, so the canonical projection is consistent with the fit.
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FactorAnalysis
from scipy.spatial import procrustes as _scipy_procrustes


def pca_neural(Y, k=15):
    """SVD-based PCA on a (T, N_units) spike-count matrix.

    Returns
    -------
    Y_pc : (T, k) projection onto top-k principal components.
    V    : (N_units, k) loadings (unit weights for each PC).
    mean : (N_units,) per-unit mean removed before projection.
    """
    # Make Y into array
    Y = np.asarray(Y, dtype=float)
    # make mean value
    mean = Y.mean(axis=0)
    # Get rid of the mean value, centered
    Yc = Y - mean
    # full_matrices=False keeps the economy SVD when T >> N.
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    # Take the most K-important
    V = Vt.T[:, :k]
    # Get the matrix PCA
    Y_pc = Yc @ V
    return Y_pc, V, mean


def trial_average_pc(Y_pc, meta, n_phase_bins=30):
    """Resample each trial's PC trajectory to ``n_phase_bins`` and average.

    Each trial's PC time series is linearly interpolated onto a common
    normalized-time grid (0 = reach start, 1 = reach end). The average across
    trials gives a single "canonical reach" in PC space for that session.

    Returns
    -------
    (n_phase_bins, k) array.
    """
    # The PCA matrix to array
    Y_pc = np.asarray(Y_pc, dtype=float)
    # Take the '15' out
    k = Y_pc.shape[1]
    # Empty for new trajectory
    trial_resampled = []
    # Group by trial_number
    for _, idx in meta.groupby("trial_number").indices.items():
        # make bin idx to an array
        idx = np.asarray(idx)
        # if too short, smaller than 3 bins
        if len(idx) < 3:
            continue
        # Make the sampling array
        t_data = np.linspace(0.0, 1.0, len(idx))
        t_targ = np.linspace(0.0, 1.0, n_phase_bins)
        # Interpolate each PC independently onto the common phase grid.
        resampled = np.column_stack([
            np.interp(t_targ, t_data, Y_pc[idx, d]) for d in range(k)
        ])
        # Put it back
        trial_resampled.append(resampled)

    if not trial_resampled:
        raise ValueError("No usable trials for trial averaging.")
    # Mean value by trials
    return np.mean(np.stack(trial_resampled, axis=0), axis=0)


def cca_align(traj_train, traj_test):
    """Fit CCA between two trial-averaged PC trajectories.

    Both inputs are (n_phase_bins, k). We use sklearn CCA with ``scale=False`` so
    that only mean centering (not variance normalization) is applied, matching
    the standard manifold-alignment formulation.

    Returns
    -------
    W_train : (k, k) rotation that takes (Y_train_pc - mean_train) into the
              shared canonical space.
    W_test  : (k, k) same for the test day.
    mean_train, mean_test : per-PC means computed from the trial-averaged
              trajectories; subtract these before applying the rotations.
    """
    # Take the 15 out
    k = traj_train.shape[1]
    # CCA from sklearn, no standardization
    cca = CCA(n_components=k, scale=False, max_iter=5000)
    cca.fit(traj_train, traj_test)
    # return two rotation matrices, also the PC value for centering
    return cca.x_rotations_, cca.y_rotations_, traj_train.mean(0), traj_test.mean(0)


def apply_alignment(Y_pc, W, mean):
    """Project a full PC time series into the canonical space defined by CCA."""
    return (np.asarray(Y_pc, dtype=float) - mean) @ W


# ---------------------------------------------------------------------------
# Sadtler/Oby/Elsayed-style manifold geometry helpers
# ---------------------------------------------------------------------------

def fa_neural(Y, k):
    """Factor analysis intrinsic manifold (Sadtler 2014; Oby 2019).

    FA models per-unit firing as ``y = L f + epsilon`` where ``f`` are k shared
    latents and ``epsilon`` is per-unit independent (private) noise. The columns
    of ``L`` (the loadings) span the *shared-variance manifold* — the directions
    of activity that are coordinated across units, as distinct from per-unit
    noise that PCA would pick up.

    Returns
    -------
    Y_fa : (T, k) latent trajectories (per-trial factor scores).
    L    : (N_units, k) factor loadings (columns = shared manifold axes).
    mean : (N_units,) per-unit mean subtracted before fitting.
    model: fitted ``FactorAnalysis`` instance (carries ``noise_variance_``).
    """
    Y = np.asarray(Y, dtype=float)
    fa = FactorAnalysis(n_components=k, max_iter=2000, random_state=0)
    Y_fa = fa.fit_transform(Y)
    L = fa.components_.T  # sklearn stores loadings as components_ shape (k, N)
    mean = fa.mean_
    return Y_fa, L, mean, fa


def participation_ratio(C):
    """Effective dimensionality of a covariance matrix.

    PR = (sum eigvals)^2 / sum(eigvals^2). Equals N for a perfectly isotropic
    distribution and 1 for a rank-1 distribution. Useful as a coordinate-free
    "how many dimensions does this manifold actually use" number.
    """
    C = np.asarray(C, dtype=float)
    eig = np.linalg.eigvalsh(C)
    eig = eig[eig > 0]
    if eig.size == 0:
        return 0.0
    s1 = eig.sum()
    s2 = (eig ** 2).sum()
    return float(s1 * s1 / s2)


def dim_for_variance(C, fraction=0.9):
    """Minimum number of components needed to explain ``fraction`` of variance."""
    C = np.asarray(C, dtype=float)
    eig = np.linalg.eigvalsh(C)[::-1]
    eig = eig[eig > 0]
    if eig.size == 0:
        return 0
    cum = np.cumsum(eig) / eig.sum()
    return int(np.searchsorted(cum, fraction) + 1)


def top_d_basis(C, d):
    """Return the top-d eigenvector basis (columns) of a covariance matrix."""
    C = np.asarray(C, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(C)
    # eigh returns ascending; take the last d for descending order.
    idx = np.argsort(eigvals)[::-1][:d]
    return eigvecs[:, idx], eigvals[idx]


def alignment_index(C_target, basis_d):
    """Elsayed 2016 normalized alignment index.

    Question: "How much of ``C_target``'s variance fits inside the subspace
    spanned by ``basis_d``, relative to the best possible d-dim subspace for
    ``C_target`` itself?"

    Returns a value in [0, 1]:
      1 = ``basis_d`` captures ``C_target`` as well as its own top-d subspace
      0 = ``basis_d`` is orthogonal to ``C_target``'s used directions

    Parameters
    ----------
    C_target : (k, k) covariance matrix of the activity we want to capture.
    basis_d  : (k, d) orthonormal basis of the candidate subspace.
    """
    C_target = np.asarray(C_target, dtype=float)
    B = np.asarray(basis_d, dtype=float)
    captured = float(np.trace(B.T @ C_target @ B))
    eig = np.linalg.eigvalsh(C_target)[::-1]
    d = B.shape[1]
    max_possible = float(eig[:d].sum())
    if max_possible <= 0:
        return np.nan
    return captured / max_possible


def outside_manifold_variance(C_target, basis_d):
    """Fraction of ``C_target``'s variance that falls *outside* ``basis_d``.

    Oby-style "new pattern" metric: if R2's covariance projects mostly into the
    R1 basis, this is small; if R2 has activity in directions orthogonal to R1,
    this grows. Returns a value in [0, 1].
    """
    C_target = np.asarray(C_target, dtype=float)
    B = np.asarray(basis_d, dtype=float)
    total = float(np.trace(C_target))
    if total <= 0:
        return np.nan
    inside = float(np.trace(B.T @ C_target @ B))
    return max(0.0, 1.0 - inside / total)


def random_alignment_null(C_target, d, n_iters=200, rng=None):
    """Distribution of alignment index for random d-dim subspaces.

    For a k-dim space, a uniformly-random d-dim subspace yields an
    *unnormalized* captured fraction ~ d/k. The normalized index (vs. top-d
    of C_target) is a stricter null — closer to d * mean(eig)/sum(top-d eig).
    We just sample and return the empirical distribution.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    C_target = np.asarray(C_target, dtype=float)
    k = C_target.shape[0]
    eig = np.linalg.eigvalsh(C_target)[::-1]
    max_possible = float(eig[:d].sum())
    if max_possible <= 0:
        return np.array([])
    out = np.empty(n_iters)
    for i in range(n_iters):
        G = rng.standard_normal((k, d))
        Q, _ = np.linalg.qr(G)
        out[i] = float(np.trace(Q.T @ C_target @ Q)) / max_possible
    return out


def procrustes_distance(X, Y):
    """Procrustes distance between two matched trajectories (rows = time).

    Wraps ``scipy.spatial.procrustes``: removes translation, isotropic scale,
    and rotation, then returns the residual sum of squared distances between
    the standardized pair (lower = more geometrically similar). Both arrays
    must have the same shape ``(n_points, n_dims)``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"Procrustes: shape mismatch {X.shape} vs {Y.shape}")
    _, _, disparity = _scipy_procrustes(X, Y)
    return float(disparity)


def canonical_correlations(traj_train, traj_test):
    """Return the canonical correlation values (length k) for diagnostic plotting.

    Reports how well the two days' PC manifolds can be aligned. Values close to
    1 across the top components mean the manifolds are nearly congruent; rapidly
    decreasing values mean only a low-dimensional subspace is shared.
    """
    # W_tr: train side CCA rotation
    # W_te: test side CCA rotation
    # m_tr: train trajectory mean
    # m_te: test trajectory mean

    W_tr, W_te, m_tr, m_te = cca_align(traj_train, traj_test)
    # Project train trajectory to canonical space
    tr_c = (traj_train - m_tr) @ W_tr
    te_c = (traj_test - m_te) @ W_te
    return np.array([
        # Calculate corr for each canonical dimension
        np.corrcoef(tr_c[:, d], te_c[:, d])[0, 1] for d in range(tr_c.shape[1])
    ])


def heldout_canonical_correlations(cache_a, cache_b, n_phase_bins=30,
                                   n_repeats=10, seed=0):
    """Held-out canonical correlations between two days' PC manifolds.

    WARNING about the in-sample `canonical_correlations` above: it fits CCA and scores
    it on the *same* trial-averaged trajectory. With ~30 phase bins and ~15 PCs that is
    over-determined and **saturates to ~1.0 for any pair** (even unrelated data), so its
    absolute values are not interpretable. This held-out version fits CCA on one random
    trial-half's averaged trajectory and scores the canonical correlation on the *other*
    half (averaged over `n_repeats` splits); it collapses to a noise floor past the
    genuinely-shared dimensions and is the metric to use for cross-day similarity.

    `cache_a` / `cache_b` must contain trial-level ``"Y_pc"`` and ``"meta"``.
    Returns a length-k vector of held-out canonical correlations.
    """
    rng = np.random.default_rng(seed)

    def traj_half(cache, trials):
        m = cache["meta"]["trial_number"].isin(trials).to_numpy()
        return trial_average_pc(cache["Y_pc"][m],
                                cache["meta"][m].reset_index(drop=True),
                                n_phase_bins=n_phase_bins)

    ta = np.array(sorted(cache_a["meta"]["trial_number"].unique()))
    tb = np.array(sorted(cache_b["meta"]["trial_number"].unique()))
    k = np.asarray(cache_a["Y_pc"]).shape[1]
    ccs = []
    for _ in range(n_repeats):
        pa, pb = rng.permutation(ta), rng.permutation(tb)
        ha, hb = len(pa) // 2, len(pb) // 2
        if min(ha, len(pa) - ha, hb, len(pb) - hb) < 2:
            continue
        try:
            W_a, W_b, m_a, m_b = cca_align(traj_half(cache_a, pa[:ha]),
                                           traj_half(cache_b, pb[:hb]))
            ca = (traj_half(cache_a, pa[ha:]) - m_a) @ W_a
            cb = (traj_half(cache_b, pb[hb:]) - m_b) @ W_b
            ccs.append(np.array([np.corrcoef(ca[:, d], cb[:, d])[0, 1]
                                 for d in range(ca.shape[1])]))
        except Exception:
            continue
    return np.nanmean(np.vstack(ccs), axis=0) if ccs else np.full(k, np.nan)
