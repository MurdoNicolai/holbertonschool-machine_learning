#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np


def pdf(X, m, S):
    """calculates the pdf"""

    if not isinstance(X, np.ndarray):
        return None

    if not isinstance(m, np.ndarray):
        return None

    if not isinstance(S, np.ndarray):
        return None

    if not S.ndim == X.ndim == 2 and m.ndim == 1:
        return None

    if not (S.shape[0] == S.shape[1] == m.shape[0] == X.shape[1]):
        return None

    d = m.shape[0]

    det_S = np.linalg.det(S)

    if det_S == 0:
        print("Error: The covariance matrix is singular.")
        return None

    inv_S = np.linalg.inv(S)

    centered_X = X - m

    exponent = -0.5 * np.sum(centered_X @ inv_S * centered_X, axis=1)

    norm_term = 1 / ((2 * np.pi) ** (d / 2) * det_S ** 0.5)

    P = norm_term * np.exp(exponent)

    P = np.maximum(P, 1e-300)

    return P
