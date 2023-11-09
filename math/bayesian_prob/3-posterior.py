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


def intersection(x, n, P, Pr):
    """return intersection of x positive in n event for each P probability"""

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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    for pr in Pr:
        if not isinstance(pr, np.float64):
            raise ValueError("All values in Pr must be in the range [0, 1]")
        if not (0 <= pr <= 1):
            raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    intersection_values = np.array([likelihood(x, n, np.array([p]))[0] * pr
                                    for p, pr in zip(P, Pr)])

    return intersection_values

def marginal(x, n, P, Pr):
    """calculates the marginal probability"""
    return (np.sum([intersection(x, n, P, Pr_i) for Pr_i in Pr]))

def posterior(x, n, P, Pr):
    """calculates the posterior probability"""

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

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    for pr in Pr:
        if not isinstance(pr, np.float64):
            raise ValueError("All values in Pr must be in the range [0, 1]")
        if not (0 <= pr <= 1):
            raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    marginal_prob = marginal(x, n, P, Pr)
    return (np.array([intersection(x, n, P, Pr_i)
                      / marginal_prob for Pr_i in Pr]))
