#!/usr/bin/env python3
"""contains functions necessary for markov chains"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    performs the Baum-Welch algorithm for a hidden markov model
    Observation -> numpy.ndarray of shape (T,) that
        contains the index of the observation

    Emission -> numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state

    Transition ->2D numpy.ndarray of shape (N, N)
        containing the transition probabilities

    Initial -> numpy.ndarray of shape (N, 1) containing the
        probability of starting in a particular hidden state

    Iterations -> the number of times expectation-max should be performed
    """
    T = len(Observations)
    M, N = Emission.shape

    for iteration in range(iterations):
        # E-step
        alpha = forward(Observations, Transition, Emission, Initial)
        beta = backward(Observations, Transition, Emission)
        xi = np.zeros((M, M, T-1))
        gamma = np.zeros((M, T))

        for t in range(T-1):
            denominator = np.sum(alpha[:, t] * beta[:, t])
            for i in range(M):
                for j in range(M):
                    xi[i, j, t] = (alpha[i, t] * Transition[i, j] *
                                   Emission[j, Observations[t+1]] *
                                   beta[j, t+1]) / denominator

            gamma[:, t] = np.sum(xi[:, :, t], axis=1)

        gamma[:, -1] = alpha[:, -1] / np.sum(alpha[:, -1])

        # M-step
        for i in range(M):
            Initial[i] = gamma[i, 0]
            for j in range(M):
                Transition[i, j] = np.sum(xi[i, j, :]) / np.sum(gamma[i, :-1])

        for j in range(M):
            for k in range(N):
                numer = np.sum((Observations == k) * gamma[j, :])
                denom = np.sum(gamma[j, :])
                Emission[j, k] = numer / denom

    return Transition, Emission


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
    return(F[1:].T)


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

    return(B[1:].T)
