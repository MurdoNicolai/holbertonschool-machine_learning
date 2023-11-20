#!/usr/bin/env python3
"""contains functions necessary for markov chains"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain::

    P -> is a square 2D numpy.ndarray of shape (n, n)
      representing the transition matrix
    """
    if not isinstance(P, np.ndarray):
        return None

    if P.ndim != 2:
        return None

    if np.sum(P, axis=0).all() != 1:
        return None

    s = np.ones((1, P.shape[0])) * (1/P.shape[0])
    P1 = P
    P2 = P @ P
    i = 10
    while not np.array_equal(P1, P2):
        if (np.isnan(s).any()):
            return None
        i -= 1
        if i == 0:
            if not(np.all(P2)):
                return None
        P1 = P2
        P2 = P2 @ P

    if not(np.all(P2)):
        return None

    sPrev = s
    s = s @ P
    while not np.array_equal(s, sPrev):
        if (np.isnan(s).any()):
            return None
        sPrev = s
        s = s @ P

    return(s)
