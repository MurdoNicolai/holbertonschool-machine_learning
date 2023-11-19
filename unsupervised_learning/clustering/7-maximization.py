#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np

def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM."""
    if not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray):
        return None, None, None

    if not X.ndim == 2 or not g.ndim == 2:
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    # Check if the sum of responsibilities for each cluster is close to the number of data points
    if not np.allclose(np.sum(g, axis=1), np.ones(k) * n):
        print("Error: The sum of responsibilities for each cluster is not close to the number of data points.")
        return None, None, None

    # Update priors (pi)
    pi = np.sum(g, axis=1) / n

    # Update means (m)
    m = (g @ X) / np.sum(g, axis=1)[:, np.newaxis]

    # Update covariances (S)
    S = np.zeros((k, d, d))

    for i in range(k):
        centered_X = X - m[i]
        S[i] = np.dot(g[i] * centered_X.T, centered_X) / np.sum(g[i])

    return pi, m, S
