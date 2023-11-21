#!/usr/bin/env python3
"""contains functions necessary for markov chains"""
import numpy as np


def absorbing(P):
    """
    determines if a markov chain is absorbing:

    P -> is a square 2D numpy.ndarray of shape (n, n)
      representing the transition matrix
    """
    if not isinstance(P, np.ndarray):
        return None

    if P.ndim != 2:
        return None

    if np.sum(P, axis=0).all() != 1:
        return None

    istrue = True
    for i in range(P.shape[0]):
        istrue = isabsorbing(P, i) and istrue
    return(istrue)


def isabsorbing(P, state, visitedstatelist=[]):
    """ determines if a state is absorbing or leads to an abosrobing state"""
    if P[state][state] == 1:
        return True
    visitedstatelist = visitedstatelist.copy()
    visitedstatelist.append(state)
    for nextState in range(P.shape[0]):
        if nextState not in visitedstatelist:
            if isabsorbing(P, nextState, visitedstatelist):
                return True
    return False
