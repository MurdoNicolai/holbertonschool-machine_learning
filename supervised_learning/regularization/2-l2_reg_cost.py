#!/usr/bin/env python3
"""containst reg_cost functino"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ calculates the cost of a neural network with L2 regularization"""
    A = cache["A{}".format(len(cache) - 1)]
    da = -(Y/A)+((1-Y)/(1-A))
    len_cache = len(cache)
    print(L, len_cache)
    newweights = {}
    for i in range(len_cache - 1, 0, -1):
        A2 = cache["A{}".format(i)]
        A1 = cache["A{}".format(i - 1)]
        dg2 = 1 - A2 ** 2
        dZ = (da) * dg2
        dW = np.matmul(dZ, A1.T)/len(A1[0])
        db = np.resize(np.sum(dZ, axis=1), (len(A2), 1))/(len(A1[0]))
        da = np.matmul(weights["W{}".format(i)].T, dZ)
        newweights.update({"W{}".format(i):
                          weights["W{}".format(i)] - alpha * dW})
        newweights.update({"b{}".format(i):
                          weights["b{}".format(i)] - alpha * db})
    weights = newweights
