#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM"""

    if (not isinstance(X, np.ndarray) or not isinstance(pi, np.ndarray) or
        not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray)):
        return None, None

    if (not X.ndim == 2 or not pi.ndim == 1 or
        not m.ndim == 2 or not S.ndim == 3):
        return None, None

    if (not (pi.shape[0] == m.shape[0] == S.shape[0] and
             m.shape[1] == X.shape[1] == S.shape[1] == S.shape[2])):
        return None, None

    k, _ = m.shape
    n, d = X.shape

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        if P is None:
            return None, None
        g[i, :] = pi[i] * P

    total_likelihood = np.sum(np.log(np.sum(g, axis=0)))

    g /= np.sum(g, axis=0)

    return g, total_likelihood
