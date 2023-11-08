#!/usr/bin/env python3
"""Contains some basic maths probability functions"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining severe side effects data given
    various hypothetical probabilities.

    Args:
        x (int): Number of patients who developed severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): Array of hypothetical probabilities
        of developing severe side effects.

    Returns:
        numpy.ndarray: Array of corresponding likelihood values.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    VE = "x must be an integer that is greater than or equal to 0"
    if not isinstance(x, int) or x < 0:
        raise ValueError(VE)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not all((0 <= p <= 1) for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    likelihood_values = np.zeros_like(P)
    for i, p in enumerate(P):
        likelihood_values[i] = (np.power(p, x) * np.power((1 - p), (n - x)))

    return likelihood_values
