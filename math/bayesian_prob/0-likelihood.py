#!/usr/bin/env python3
"""Contains some basic maths probability functions"""
import numpy as np


def binomial_coefficient(n, k):
    """Calculate n! / (k! * (n - k)!)"""
    return (np.math.factorial(n) // (np.math.factorial(k) *
                                     np.math.factorial(n - k)))


def likelihood(x, n, P):
    """return likelihood of x positive in n event for each P probability"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    VE = "x must be an integer that is greater than or equal to 0"
    if not isinstance(x, int) or x < 0:
        raise ValueError(VE)
    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not all(0 <= p <= 1 for p in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the binomial probability mass function
    likelihoods = np.array([binomial_coefficient(n, x) * (p ** x) *
                            ((1 - p) ** (n - x)) for p in P])

    return likelihoods
