#!/usr/bin/env python3
"""creates a class for Hyperparameter Tuning"""
import numpy as np


class GaussianProcess:
    """represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
            X_init -> numpy.ndarray: inputs
            Y_init -> numpy.ndarray: outputs each input in X_init
            l -> length of the kernel
            sigma_f -> standard deviation of output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = sigma_f**2 * self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix between two matrices"""
        new_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        div = -2*(self.l**2)
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                new_matrix[i][j] = np.exp(np.linalg.norm(X1[i]-X2[j])**2/div)

        return(new_matrix)
