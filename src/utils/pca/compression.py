"""PCA image compression helpers for the MNIST 0/1/2 study.

PCA gives a lossy image codec: keep only the top `k` principal components, and
each 784-pixel image is stored as just `k` numbers (its "code"). Reconstruction
maps those `k` codes back into pixel space. Fewer components -> smaller codes
(higher compression ratio) but blurrier reconstructions -> this module exposes
the pieces needed to study that quality-vs-size trade-off.

Data contract comes from `utils.mnist_data` (X is (n, 784) float32 in [0, 1]).

Importable helpers (used by `src/pca/compression.ipynb`):
    - fit_pca(X, n_components):      fit a PCA on a design matrix
    - compress(pca, X):             images -> low-dim codes
    - reconstruct(pca, codes):      codes -> reconstructed images in [0, 1]
    - reconstruction_error(X, Xhat):per-pixel mean squared error
    - compression_ratio(k):         784 / k, the storage saving factor
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from utils.mnist_data import N_FEATURES


def fit_pca(X, n_components: int) -> PCA:
    """Fit a PCA model with `n_components` on the design matrix `X`.

    Returns
    -------
    sklearn.decomposition.PCA
        A fitted PCA (mean-centered internally) ready for `compress`/`reconstruct`.
    """
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X)
    return pca


def compress(pca: PCA, X):
    """Project images onto the top principal components (the encode step).

    Returns
    -------
    np.ndarray, shape (n_samples, n_components)
        The low-dimensional codes -- this is all that must be stored per image.
    """
    return pca.transform(X)


def reconstruct(pca: PCA, codes):
    """Map low-dim codes back into image space (the decode step).

    The inverse transform can produce values slightly outside [0, 1], so the
    result is clipped back to the valid pixel range.

    Returns
    -------
    np.ndarray, shape (n_samples, 784), float in [0, 1]
        The reconstructed images.
    """
    X_hat = pca.inverse_transform(codes)
    return np.clip(X_hat, 0.0, 1.0)


def reconstruction_error(X, X_hat) -> float:
    """Per-pixel mean squared error between originals and reconstructions.

    Returns
    -------
    float
        Mean over all samples and pixels of (X - X_hat)**2. Lower is better.
    """
    X = np.asarray(X, dtype=np.float64)
    X_hat = np.asarray(X_hat, dtype=np.float64)
    return float(np.mean((X - X_hat) ** 2))


def compression_ratio(n_components: int, n_features: int = N_FEATURES) -> float:
    """Storage saving factor from keeping `n_components` instead of every pixel.

    Returns
    -------
    float
        `n_features / n_components` (e.g. 784 / 20 = 39.2x smaller codes).
    """
    return n_features / n_components


if __name__ == "__main__":
    from utils.mnist_data import load_digits

    X, _ = load_digits(per_class=1000)
    print(f"Loaded X {X.shape}")
    for k in (10, 50, 100):
        pca = fit_pca(X, k)
        codes = compress(pca, X)
        X_hat = reconstruct(pca, codes)
        mse = reconstruction_error(X, X_hat)
        ratio = compression_ratio(k)
        print(
            f"k={k:>3}  compression_ratio={ratio:6.2f}x  "
            f"reconstruction_MSE={mse:.5f}  codes_shape={codes.shape}"
        )
