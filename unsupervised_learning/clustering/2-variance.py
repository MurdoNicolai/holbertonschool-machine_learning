
#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np


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


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""

    if not isinstance(X, np.ndarray):
        return (None, None)

    if X.ndim != 2:
        return (None, None)

    if not isinstance(k, int) or k <= 0:
        return (None, None)

    if not isinstance(iterations, int) or iterations <= 0:
        return (None, None)
    n, d = X.shape

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    C = initialize(X, k)

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        new_C = np.array([X[clss == i].mean(axis=0)
                          if np.sum(clss == i) > 0
                          else np.random.uniform(low=min_vals,
                                                 high=max_vals, size=(d,))
                          for i in range(k)])

        if np.array_equal(C, new_C):
            return C, clss

        C = new_C

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    return C, clss


def variance(X, C):
    """calculates the total intra-cluster variance for a data set"""
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    var = np.sum((X - C[clss]) ** 2)
    return (var)
