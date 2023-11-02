#!/usr/bin/env python3
"""contains a mean calculator"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set"""
    mean = np.mean(X, axis=0, keepdims=True)
    centered_data = X - mean
    cov = np.dot(centered_data.T, centered_data) / (X.shape[0] - 1)
    return mean.T, cov


class MultiNormal:
    """ represents a Multivariate Normal distribution:"""

    def __init__(self, data):
        """
            data is a numpy.ndarray of shape (d, n) containing the data set:
            n is the number of data points
            d is the number of dimensions in each data point
        """
        if type(data) is not np.ndarray or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean, self.cov = mean_cov(data.T)
