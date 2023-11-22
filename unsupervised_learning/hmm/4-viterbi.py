#!/usr/bin/env python3
"""contains functions necessary for markov chains"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    calculate the most likely sequence of hidden states of hidden markov model:
    Observation -> numpy.ndarray of shape (T,) that
        contains the index of the observation

    Emission -> numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state

    Transition ->2D numpy.ndarray of shape (N, N)
        containing the transition probabilities

    Initial -> numpy.ndarray of shape (N, 1) containing the
        probability of starting in a particular hidden state
    """
    L_PathChances = Initial.T[0] * Emission.T[Observation[0]]
    L_Paths = np.ones((Initial.shape[0], Observation.shape[0]), dtype=int) * -1
    for nb_obs in range(1, Observation.shape[0]):
        temp_LPC = L_PathChances.copy()
        for path in range(Transition.shape[0]):
            L_PathChances[path] = np.max(Transition[:, path] * temp_LPC)
            L_Paths[path, nb_obs] = np.argmax(Transition[:, path] * temp_LPC)
        L_PathChances = L_PathChances * Emission.T[Observation[nb_obs]]

    best_last_state = np.argmax(L_PathChances)
    best_path = [best_last_state]
    for t in range(Observation.shape[0]-1, 0, -1):
        best_last_state = L_Paths[best_last_state, t]
        best_path.insert(0, best_last_state)
    return np.array(best_path), np.max(L_PathChances)
