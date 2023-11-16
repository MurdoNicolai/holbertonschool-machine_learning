#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ initializes variables for a Gaussian Mixture Model:"""

    if not isinstance(X, np.ndarray):
        return (None, None, None)

    if X.ndim != 2:
        return (None, None, None)

    if not isinstance(k, int) or k <= 0:
        return (None, None, None)

    n, d = X.shape

    pi = np.full(k, 1/k)

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    S = np.array([np.identity(d)] * k)

    return pi, m, S
