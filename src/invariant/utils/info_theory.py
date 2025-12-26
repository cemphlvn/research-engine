"""Information-theoretic primitives for invariance quantification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors


def entropy(x: NDArray[np.float64], bins: int = 30) -> float:
    """Compute entropy of a distribution.

    Args:
        x: Input array (1D for scalar, 2D for multivariate)
        bins: Number of bins for discretization

    Returns:
        Entropy in nats
    """
    if x.ndim == 1:
        if len(x) < 2 or np.std(x) < 1e-10:
            return 0.0  # Constant or insufficient data

        # Use counts then normalize to avoid density=True warnings
        hist, _ = np.histogram(x, bins=bins)
        hist = hist.astype(np.float64)
        if hist.sum() < 1e-10:
            return 0.0
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return float(scipy_entropy(hist))

    # For multivariate, use sum of marginal entropies (upper bound)
    return float(sum(entropy(x[:, i], bins) for i in range(x.shape[1])))


def mutual_information(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    bins: int = 30,
    method: str = "binned",
) -> float:
    """Compute mutual information I(X;Y).

    Args:
        x: First variable
        y: Second variable
        bins: Number of bins for discretization
        method: "binned" or "knn" (Kraskov estimator)

    Returns:
        Mutual information in nats
    """
    if method == "binned":
        # Discretize continuous variables
        x_discrete = np.digitize(x.ravel(), np.linspace(x.min(), x.max(), bins))
        y_discrete = np.digitize(y.ravel(), np.linspace(y.min(), y.max(), bins))
        return float(mutual_info_score(x_discrete, y_discrete))

    elif method == "knn":
        # Kraskov estimator for continuous variables
        return _kraskov_mi(x, y)

    raise ValueError(f"Unknown method: {method}")


def _kraskov_mi(x: NDArray[np.float64], y: NDArray[np.float64], k: int = 3) -> float:
    """Kraskov MI estimator using k-NN.

    Reference: Kraskov et al., PRE 69, 066138 (2004)
    """
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    y = np.atleast_2d(y.T).T if y.ndim == 1 else y
    n = len(x)

    # Joint space
    xy = np.hstack([x, y])

    # Find k-th neighbor distances in joint space
    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev")
    nn_joint.fit(xy)
    distances, _ = nn_joint.kneighbors(xy)
    eps = distances[:, k]

    # Count neighbors within eps in marginal spaces
    nn_x = NearestNeighbors(metric="chebyshev")
    nn_x.fit(x)
    nn_y = NearestNeighbors(metric="chebyshev")
    nn_y.fit(y)

    nx = np.array([len(nn_x.radius_neighbors([xi], eps[i], return_distance=False)[0]) - 1
                   for i, xi in enumerate(x)])
    ny = np.array([len(nn_y.radius_neighbors([yi], eps[i], return_distance=False)[0]) - 1
                   for i, yi in enumerate(y)])

    # Digamma function
    from scipy.special import digamma
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(max(0, mi))


def kl_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    eps: float = 1e-10,
) -> float:
    """Compute KL divergence D_KL(P || Q).

    Args:
        p: Distribution P (will be normalized)
        q: Distribution Q (will be normalized)
        eps: Small value to avoid log(0)

    Returns:
        KL divergence (non-negative)
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()

    # Normalize
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)

    # Add epsilon and renormalize
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def stability_score(
    original: NDArray[np.float64],
    transformed: NDArray[np.float64],
) -> float:
    """Compute cosine similarity between mean representations.

    Args:
        original: Original embeddings (N x D)
        transformed: Transformed embeddings (N x D)

    Returns:
        Cosine similarity in [0, 1]
    """
    orig_mean = np.mean(original, axis=0)
    trans_mean = np.mean(transformed, axis=0)

    dot = np.dot(orig_mean, trans_mean)
    norm = np.linalg.norm(orig_mean) * np.linalg.norm(trans_mean)

    if norm < 1e-10:
        return 0.0

    # Map from [-1, 1] to [0, 1]
    return float((dot / norm + 1) / 2)


def info_preserved_ratio(
    original: NDArray[np.float64],
    transformed: NDArray[np.float64],
) -> float:
    """Compute information preservation between original and transformed.

    Uses average pairwise similarity preservation as a proxy for MI.
    This is more robust than binned MI for high-dimensional embeddings.

    Args:
        original: Original representations (N x D)
        transformed: Transformed representations (N x D)

    Returns:
        Ratio in [0, 1] - how much relational structure is preserved
    """
    if len(original) < 2:
        return 1.0

    # Compute pairwise similarities
    orig_sim = original @ original.T
    trans_sim = transformed @ transformed.T

    # Flatten upper triangles (excluding diagonal)
    n = len(original)
    orig_pairs = []
    trans_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            orig_pairs.append(orig_sim[i, j])
            trans_pairs.append(trans_sim[i, j])

    if len(orig_pairs) < 2:
        return 1.0

    orig_pairs = np.array(orig_pairs)
    trans_pairs = np.array(trans_pairs)

    # Correlation of pairwise similarities = structure preservation
    if np.std(orig_pairs) < 1e-10 or np.std(trans_pairs) < 1e-10:
        return 1.0 if np.allclose(orig_pairs, trans_pairs) else 0.0

    corr = np.corrcoef(orig_pairs, trans_pairs)[0, 1]

    # Map correlation [-1, 1] to [0, 1]
    return float(max(0, (corr + 1) / 2))
