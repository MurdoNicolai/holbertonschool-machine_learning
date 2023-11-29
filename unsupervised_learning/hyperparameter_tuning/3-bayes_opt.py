#!/usr/bin/env python3
"""creates a class for Hyperparameter Tuning"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """performs Bayesian optimization on a noiseless 1D Gaussian process"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
            X_init -> numpy.ndarray: inputs
            Y_init -> numpy.ndarray: outputs each input in X_init
            l -> length of the kernel
            sigma_f -> standard deviation of output
            bounds -> tuple(min, max): bounds of space to search
            ac_samples -> number of samples that should be analyzed
            xsi -> exploration-exploitation factor for acquisition
            minimize -> optimize for minimization (True) maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = np.expand_dims(self.X_s, axis=1)
        self.xsi = xsi
        self.minimize = minimize
