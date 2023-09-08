#!/usr/bin/env python3
"""containst reg_cost functino"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ gradient descends weights"""
    A = cache["A{}".format(len(cache) - 1)]
    dZ = A - Y
    newweights = {}
    for i in range(L, 0, -1):
        A_prev = cache["A{}".format(i - 1)]

        dW = np.matmul(dZ, A_prev .T)/len(A_prev[0])
        db = np.sum(dZ, axis=1, keepdims=True) / len(A_prev[0])
        da = np.matmul(weights["W{}".format(i)].T, dZ)
        newweights.update({"W{}".format(i):
                          weights["W{}".format(i)] - alpha * dW})
        newweights.update({"b{}".format(i):
                          weights["b{}".format(i)] - alpha * db})

        dg2 = 1 - A_prev ** 2
        dZ = (da) * dg2
    weights.update(newweights)
