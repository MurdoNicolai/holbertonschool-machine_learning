#!/usr/bin/env python3
"""contains a mean calculator"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set"""
    if type(X) is not np.ndarray or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    return np.mean(X, axis=0), np.cov(X.T)

