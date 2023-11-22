#!/usr/bin/env python3
"""contains functions necessary for markov chains"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model:
    Observation -> numpy.ndarray of shape (T,) that
        contains the index of the observation

    Emission -> numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state

    Transition ->2D numpy.ndarray of shape (N, N)
        containing the transition probabilities

    Initial -> numpy.ndarray of shape (N, 1) containing the
        probability of starting in a particular hidden state
    """
    Initial = Initial.T
    F = np.zeros((Initial.shape))
    for nb_observations in range(Observation.shape[0]):
        Initial = Initial * Emission.T[Observation[nb_observations]]
        F = np.vstack((F, Initial))
        Initial = Initial @ Transition
    return(np.sum(F[nb_observations + 1]), F)
