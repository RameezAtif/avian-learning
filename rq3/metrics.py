"""Metrics for RQ3: boundary strength and progressive differentiation."""

import numpy as np


def categorical_boundary_strength(h, labels):
    """
    h:      (N, hidden_dim) hidden representations
    labels: (N,) integer category labels

    Returns ratio of between-category distance to within-category distance.
    Higher = sharper boundaries.
    """
    N = h.shape[0]
    within, between = [], []

    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(h[i] - h[j])
            if labels[i] == labels[j]:
                within.append(d)
            else:
                between.append(d)

    if len(within) == 0 or len(between) == 0:
        return np.nan

    return float(np.mean(between) / (np.mean(within) + 1e-12))


def progressive_differentiation(student, X, Y):
    """
    SVD of the input-output correlation matrix Sigma = Y^T X.

    Singular values correspond to semantic modes. As training
    progresses, smaller modes emerge later (Cao's progressive
    differentiation).

    Returns the singular values as a descending array.
    """
    h, _ = student.forward(X)
    Sigma = Y.T @ h                          # (K, D_h)
    U, S, Vt = np.linalg.svd(Sigma, full_matrices=False)
    return S