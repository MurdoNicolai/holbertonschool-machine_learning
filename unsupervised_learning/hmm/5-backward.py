#!/usr/bin/env python3
"""contains functions necessary for markov chains"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    Observation -> numpy.ndarray of shape (T,) that
        contains the index of the observation

    Emission -> numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state

    Transition ->2D numpy.ndarray of shape (N, N)
        containing the transition probabilities

    Initial -> numpy.ndarray of shape (N, 1) containing the
        probability of starting in a particular hidden state
    """
    N = Initial.shape[0]
    Initial = Initial.T
    F = np.zeros((1, N))
    for nb_observations in range(Observation.shape[0]):
        Initial = Initial * Emission.T[Observation[nb_observations]]
        F = np.vstack((F, Initial))
        Initial = Initial @ Transition
    P = np.sum(F[nb_observations + 1])

    B = np.ones((F.shape))
    for nb_obs in range(Observation.shape[0]-1, -1, - 1):
        for s in range(N):
            B[nb_obs][s] = (((B[nb_obs][s] * Transition[s]) *
                            B[nb_obs + 1]) @ Emission.T[Observation[nb_obs]])

    return(P, B[1:].T)
