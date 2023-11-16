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
    C, clss = kmeans(X, kmin, iterations)
    results = [C]
    first_variance = variance(X, C)
    d_vars = [0]
    if kmax is None:
        kmax = kmin + 2
    for k in range(kmin + 1, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append(C)
        d_vars.append(first_variance - variance(X, C))
    return results, d_vars
