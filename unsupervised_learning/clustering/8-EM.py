#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np
from math import isclose
from typing import Optional, Tuple


def expectation_maximization(X: np.ndarray, k: int, iterations: int = 1000, tol: float = 1e-5, verbose: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Performs the Expectation-Maximization algorithm for a Gaussian Mixture Model (GMM).

    Args:
        X (np.ndarray): The data set of shape (n, d).
        k (int): The number of clusters.
        iterations (int): The maximum number of iterations for the algorithm. Defaults to 1000.
        tol (float): Tolerance of the log likelihood to determine early stopping. Defaults to 1e-5.
        verbose (bool): If True, print information about the algorithm. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: The priors (pi), centroid means (m),
        covariance matrices (S), probabilities (g), and log likelihood (l).
        Returns None, None, None, None, None on failure.
    """

    # Import necessary functions
    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    # Initialize parameters
    pi, m, S = initialize(X, k)
    prev_likelihood = float('-inf')

    # Main EM loop
    for i in range(iterations):
        # Expectation step
        g, likelihood = expectation(X, pi, m, S)

        if g is None or likelihood is None:
            return None, None, None, None, None

        # Maximization step
        pi, m, S = maximization(X, g)

        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Check for convergence
        if isclose(likelihood, prev_likelihood, abs_tol=tol):
            break

        prev_likelihood = likelihood

        # Print verbose information
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {round(likelihood, 5)}")

    # Print final iteration if verbose is True
    if verbose:
        print(f"Log Likelihood after {iterations} iterations: {round(likelihood, 5)}")

    return pi, m, S, g, likelihood
