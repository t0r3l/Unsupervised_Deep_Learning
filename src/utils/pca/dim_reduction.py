"""PCA dimensionality-reduction helpers for the MNIST 0/1/2 study.

Principal Component Analysis rotates the 784-dimensional pixel space into an
orthogonal basis ordered by how much variance each axis explains. This module
implements PCA from scratch -- mean-centering, a covariance matrix, and an
eigendecomposition -- rather than calling `sklearn.decomposition.PCA`, so every
step of the algorithm stays visible. It backs `src/pca/dim_reduction.ipynb`:

    - fit_pca(X, n_components):            fit a PCA model on a design matrix
    - cumulative_explained_variance(pca): running sum of explained-variance ratio
    - n_components_for_variance(X, thr):   #components needed to reach a variance
    - project(pca, X) / project_2d(X):     map images onto principal components

All functions follow the shared data contract from `utils.mnist_data`: X is a
(n_samples, 784) float32 matrix in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCA:
    """A fitted PCA model: the mean, principal axes, and their variances.

    Exposes the handful of `sklearn.decomposition.PCA`-shaped attributes the
    rest of this project relies on (`n_components_`, `explained_variance_ratio_`,
    `.transform(...)`), even though the fit below never touches sklearn.
    """

    mean_: np.ndarray                      # (n_features,) subtracted before projecting
    components_: np.ndarray                # (n_components, n_features) axes, descending variance
    explained_variance_: np.ndarray        # (n_components,) eigenvalue (variance) of each axis
    explained_variance_ratio_: np.ndarray  # (n_components,) fraction of total variance per axis

    @property
    def n_components_(self) -> int:
        return self.components_.shape[0]

    def transform(self, X) -> np.ndarray:
        """Project `X` onto the fitted principal axes."""
        Xc = np.asarray(X) - self.mean_
        return Xc @ self.components_.T


def fit_pca(X, n_components=None) -> PCA:
    """Fit a PCA model on the design matrix `X` by decomposing it step by step.

    1. Center `X` on its per-feature mean -- PCA looks for directions of
       variance *around* the mean.
    2. Build the covariance matrix of the centered features.
    3. Eigendecompose that (symmetric) covariance matrix: eigenvectors are the
       principal axes, eigenvalues are the variance each axis explains.
    4. Sort the eigenpairs by descending eigenvalue.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Flat, scaled images (the `utils.mnist_data` contract).
    n_components : int, float or None
        None keeps every axis (== n_features here, since MNIST has far more
        samples than pixels); an int keeps that many leading axes; a float in
        (0, 1] keeps just enough leading axes to explain that fraction of
        variance.

    Returns
    -------
    PCA
        The fitted model (call `.transform(...)` to project new data).
    """
    X = np.asarray(X, dtype=np.float64)

    # Step 1: center.
    mean_ = X.mean(axis=0)
    Xc = X - mean_

    # Step 2: covariance matrix of the centered features, shape (n_features, n_features).
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)

    # Step 3: eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Step 4: sort descending -- largest-variance axis first. Clip float noise
    # that can push a near-zero eigenvalue slightly below 0.
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    ratio = eigvals / eigvals.sum()

    # Step 5: decide how many leading axes to keep.
    if n_components is None:
        k = len(eigvals)
    elif isinstance(n_components, float):
        if not 0.0 < n_components <= 1.0:
            raise ValueError(f"n_components float must be in (0, 1], got {n_components!r}")
        k = int(np.searchsorted(np.cumsum(ratio), n_components) + 1)
    else:
        k = int(n_components)

    return PCA(
        mean_=mean_,
        components_=eigvecs[:, :k].T,
        explained_variance_=eigvals[:k],
        explained_variance_ratio_=ratio[:k],
    )


def cumulative_explained_variance(pca: PCA) -> np.ndarray:
    """Running total of the explained-variance ratio over the components.

    Returns
    -------
    np.ndarray, shape (n_components,)
        `cumsum` of `pca.explained_variance_ratio_`; entry i is the fraction of
        total variance captured by the first i + 1 principal components.
    """
    return np.cumsum(pca.explained_variance_ratio_)


def n_components_for_variance(X, threshold: float = 0.90) -> int:
    """Smallest number of principal components that reaches a variance threshold.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to fit PCA on.
    threshold : float in (0, 1]
        Fraction of total variance to reach (e.g. 0.90 for 90%).

    Returns
    -------
    int
        The minimum number of leading components whose cumulative
        explained-variance ratio is >= `threshold`.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    pca = fit_pca(X)
    cum = cumulative_explained_variance(pca)
    # +1 because argmax gives a 0-based index; we want a count of components.
    return int(np.searchsorted(cum, threshold) + 1)


def project(pca: PCA, X) -> np.ndarray:
    """Project `X` onto the principal components of a fitted `pca`.

    Returns
    -------
    np.ndarray, shape (n_samples, pca.n_components_)
        The coordinates of each row of `X` in the PCA basis.
    """
    return pca.transform(X)


def project_2d(X) -> np.ndarray:
    """Convenience: fit PCA and return the first 2 principal components of `X`.

    Handy for scatter plots. Fits a fresh 2-component PCA on `X`.

    Returns
    -------
    np.ndarray, shape (n_samples, 2)
        Coordinates on PC1 and PC2.
    """
    pca = fit_pca(X, n_components=2)
    return project(pca, X)


if __name__ == "__main__":
    from utils.mnist_data import load_digits

    X, y = load_digits(per_class=1000)
    print(f"X {X.shape} {X.dtype} range [{X.min():.2f}, {X.max():.2f}]")

    pca = fit_pca(X)
    cum = cumulative_explained_variance(pca)
    print(f"fitted PCA with {pca.n_components_} components")
    print(f"first 5 explained-variance ratios: {pca.explained_variance_ratio_[:5]}")

    for thr in (0.90, 0.95):
        k = n_components_for_variance(X, thr)
        print(f"components for {thr:.0%} variance: {k}")

    coords2d = project_2d(X)
    print(f"project_2d -> {coords2d.shape}")
