#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ initializes variables for a Gaussian Mixture Model:"""
    n, d = X.shape

    pi = np.full(k, 1/k)

    m = kmeans(X, k)
    if m is None:
        return None, None, None

    S = np.array([np.identity(d)] * k)

    return pi, m, S
