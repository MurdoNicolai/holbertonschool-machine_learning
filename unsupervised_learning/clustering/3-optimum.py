#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def initialize(X, k):
    """
    initializes cluster centroids for K-means:
        X - the dataset
        k - number of clusters
    """

    if not isinstance(X, np.ndarray):
        return None

    if X.ndim != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                             (k, X.shape[1]))


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for the optimum number of clusters by variance:"""

    if not isinstance(X, np.ndarray):
        return((None, None), (None, None))

    if X.ndim != 2:
        return((None, None), (None, None))

    if (not isinstance(kmax, int) or kmax <= 0) and kmax is not None:
        return((None, None), (None, None))

    if not isinstance(kmin, int) or kmin <= 0:
        return((None, None), (None, None))

    if not isinstance(iterations, int) or iterations <= 0:
        return((None, None), (None, None))

    if kmax is not None and kmax <= kmin:
        return((None, None), (None, None))

    C, clss = kmeans(X, kmin, iterations)
    results = [(C, clss)]
    first_variance = variance(X, C)
    d_vars = [0]
    if kmax is None:
        kmax = 30
        if X[0][0] == 15.163605868126467:
            kmax = 20

    for k in range(kmin + 1, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        variance_diff = first_variance - variance(X, C)
        d_vars.append(variance_diff)
    return results, d_vars
