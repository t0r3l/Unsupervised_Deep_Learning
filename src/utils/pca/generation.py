"""Generate synthetic MNIST digits by sampling in PCA latent space.

The idea is a simple *linear-Gaussian* generative model. We fit a PCA on the
0/1/2 digit images to get a low-dimensional latent space, model the latent
codes with a (multivariate) Gaussian, draw fresh latent vectors from that
Gaussian, and `inverse_transform` them back into 28x28 pixel space. Fitting one
Gaussian per digit class lets us generate a chosen digit on demand.

This is deliberately a baseline: PCA + a Gaussian can only capture *linear*,
*unimodal* structure, so the samples look like smooth "average" digits rather
than the crisp, varied strokes a nonlinear model (VAE/GAN) would produce.

Contract (see src/utils/mnist_data.py):
    X : np.ndarray, shape (n_samples, 784), float32 in [0, 1]
    y : np.ndarray, shape (n_samples,), int  -- digit label per row
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

IMG_SIZE = 28
N_FEATURES = IMG_SIZE * IMG_SIZE  # 784


def fit_pca(X, n_components: int = 50) -> PCA:
    """Fit a PCA that maps images (n, 784) to a `n_components`-dim latent space.

    Returns
    -------
    sklearn.decomposition.PCA : the fitted PCA (use `.transform` /
    `.inverse_transform` to move between pixel and latent space).
    """
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(X)
    return pca


def fit_latent_gaussian(pca: PCA, X, y=None, ridge: float = 1e-6):
    """Fit a Gaussian to the PCA latent codes of `X`.

    The latent codes are `pca.transform(X)`. A small `ridge` is added to the
    covariance diagonal so it stays positive-definite (numerically safe to
    sample from). If `y` is given, one Gaussian is fitted per class.

    Parameters
    ----------
    pca : fitted PCA
    X : np.ndarray, shape (n, 784)
    y : np.ndarray or None
        If given, fit a separate Gaussian per unique label.
    ridge : float
        Amount added to the covariance diagonal for stability.

    Returns
    -------
    If `y is None`: a tuple ``(mean, cov)`` with shapes (d,) and (d, d).
    Otherwise: a dict ``{label: (mean, cov)}`` mapping each class label to its
    Gaussian parameters.
    """
    Z = pca.transform(X)
    d = Z.shape[1]

    def _gaussian(codes):
        mean = codes.mean(axis=0)
        cov = np.cov(codes, rowvar=False)
        cov = np.atleast_2d(cov) + ridge * np.eye(d)
        return mean, cov

    if y is None:
        return _gaussian(Z)

    y = np.asarray(y)
    return {int(label): _gaussian(Z[y == label]) for label in np.unique(y)}


def sample_latent(mean, cov, n: int, seed: int = 0):
    """Draw `n` latent vectors from the Gaussian ``N(mean, cov)``.

    Returns
    -------
    np.ndarray, shape (n, d) : sampled latent codes.
    """
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.asarray(mean), np.asarray(cov), size=n)


def generate(pca: PCA, latents, as_images: bool = False):
    """Decode latent vectors into pixel-space images.

    The latents are pushed through `pca.inverse_transform` and clipped to the
    valid [0, 1] pixel range.

    Parameters
    ----------
    pca : fitted PCA
    latents : np.ndarray, shape (n, d)
    as_images : bool
        If True, return (n, 28, 28); otherwise flat (n, 784).

    Returns
    -------
    np.ndarray : generated images, float in [0, 1].
    """
    imgs = pca.inverse_transform(np.asarray(latents))
    imgs = np.clip(imgs, 0.0, 1.0)
    if as_images:
        return imgs.reshape(-1, IMG_SIZE, IMG_SIZE)
    return imgs


def interpolate(pca: PCA, code_a, code_b, n_steps: int = 10, as_images: bool = False):
    """Linearly walk between two latent codes and decode each step.

    Handy for visualising the latent space: pass the `pca.transform` codes of
    two real digits to morph one into the other.

    Parameters
    ----------
    pca : fitted PCA
    code_a, code_b : np.ndarray, shape (d,)
        Endpoint latent codes.
    n_steps : int
        Number of images along the path (inclusive of both endpoints).
    as_images : bool
        If True, return (n_steps, 28, 28); otherwise flat (n_steps, 784).

    Returns
    -------
    np.ndarray : the interpolated, decoded images, float in [0, 1].
    """
    code_a = np.asarray(code_a)
    code_b = np.asarray(code_b)
    ts = np.linspace(0.0, 1.0, n_steps)[:, None]
    path = (1.0 - ts) * code_a[None, :] + ts * code_b[None, :]
    return generate(pca, path, as_images=as_images)


if __name__ == "__main__":
    from utils.mnist_data import load_digits

    X, y = load_digits(per_class=300)
    pca = fit_pca(X, n_components=50)
    print(f"PCA: {pca.n_components_} comps, "
          f"explained var {pca.explained_variance_ratio_.sum():.3f}")

    # Overall Gaussian: sample a few digits.
    mean, cov = fit_latent_gaussian(pca, X)
    latents = sample_latent(mean, cov, n=5, seed=0)
    samples = generate(pca, latents)
    print(f"samples {samples.shape} {samples.dtype} "
          f"range [{samples.min():.2f}, {samples.max():.2f}]")

    # Per-class Gaussians.
    per_class = fit_latent_gaussian(pca, X, y=y)
    for label, (m, c) in per_class.items():
        z = sample_latent(m, c, n=3, seed=label)
        imgs = generate(pca, z, as_images=True)
        print(f"digit {label}: generated {imgs.shape}")

    # Interpolation between two real digits.
    codes = pca.transform(X[:2])
    walk = interpolate(pca, codes[0], codes[1], n_steps=8, as_images=True)
    print(f"interpolation {walk.shape}")
