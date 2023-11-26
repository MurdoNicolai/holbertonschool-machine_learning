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


def forward(obs, trans, emit, init):
    """
    performs forward propagation
    """
    T = len(obs)
    M = trans.shape[0]

    alpha = np.zeros((M, T))
    alpha[:, 0] = np.squeeze(init * emit[:, obs[0]])

    for t in range(1, T):
        for j in range(M):
            alpha[j, t] = emit[j, obs[t]] * np.sum(alpha[:, t-1] * trans[:, j])

    return alpha


def backward(obs, trans, emit):
    """
    performs backward propagation
    """
    T = len(obs)
    M = trans.shape[0]

    beta = np.zeros((M, T))
    beta[:, -1] = 1

    for t in range(T-2, -1, -1):
        for i in range(M):
            beta[i, t] = np.sum(beta[:, t+1] * trans[i, :] * emit[:, obs[t+1]])

    return beta
