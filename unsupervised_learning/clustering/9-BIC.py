#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using the Bayesian Information Criterion (BIC).

    Args:
        X (np.ndarray): The data set of shape (n, d).
        kmin (int): The minimum number of clusters to check for (inclusive). Defaults to 1.
        kmax (int): The maximum number of clusters to check for (inclusive). If None, set to the maximum possible clusters. Defaults to None.
        iterations (int): The maximum number of iterations for the EM algorithm. Defaults to 1000.
        tol (float): Tolerance for the EM algorithm. Defaults to 1e-5.
        verbose (bool): If True, the EM algorithm prints information to the standard output. Defaults to False.

    Returns:
        Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
            best_k: The best value for k based on its BIC.
            best_result: Tuple containing pi, m, S for the best number of clusters.
                pi is a numpy.ndarray of shape (k,) containing the cluster priors for the best number of clusters.
                m is a numpy.ndarray of shape (k, d) containing the centroid means for the best number of clusters.
                S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for the best number of clusters.
            l: A numpy.ndarray of shape (kmax - kmin + 1) containing the log likelihood for each cluster size tested.
            b: A numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value for each cluster size tested.
    """
    expectation_maximization = __import__('8-EM').expectation_maximization

    if kmax is None:
        kmax = X.shape[0]

    if kmin <= 0 or kmax <= 0 or kmin > kmax:
        return None, None, None, None

    l_values = []
    b_values = []

    best_k = None
    best_result = None
    best_bic = float('-inf')

    for k in range(kmin, kmax + 1):
        result = expectation_maximization(X, k, iterations=iterations, tol=tol, verbose=verbose)

        if result is None:
            return None, None, None, None

        pi, m, S, g, likelihood = result

        # Calculate the number of parameters (p)
        p = k * (1 + X.shape[1] + X.shape[1] * (X.shape[1] + 1) // 2)

        # Calculate BIC
        bic = p * np.log(X.shape[0]) - 2 * likelihood

        l_values.append(likelihood)
        b_values.append(bic)

        if bic > best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    l_values = np.array(l_values)
    b_values = np.array(b_values)

    return best_k, best_result, l_values, b_values
