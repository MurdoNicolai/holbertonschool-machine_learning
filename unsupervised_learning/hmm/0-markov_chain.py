#!/usr/bin/env python3
"""contains functions necessary for markov chains"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    determines the probability of a markov chain being in a
    particular state after a specified number of iterations:

    P -> is a square 2D numpy.ndarray of shape (n, n)
      representing the transition matrix

    s -> is a numpy.ndarray of shape (1, n) representing the probability of
      starting in each state
    t -> is the number of iterations that the markov chain has been through
    """

    if not isinstance(P, np.ndarray):
        return None

    if P.ndim != 2:
        return None

    if not isinstance(s, np.ndarray):
        return None

    if s.ndim != 2:
        return None

    if not isinstance(t, int):
        return None

    if t < 1:
        return None

    if np.sum(P, axis=0).all() != 1 or np.sum(s) != 1:
        return None

    for iter in range(t):
        s = s @ P

    return(s)
