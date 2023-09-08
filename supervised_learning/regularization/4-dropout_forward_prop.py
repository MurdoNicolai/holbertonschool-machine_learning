#!/usr/bin/env python3
""" contains dropout_forward_prop"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conduct forward propagation using Dropout.
    """

    cache = {}
    A = X
    w_nb = 0
    while w_nb < L:
        w_nb = w_nb + 1
        Z = (np.matmul(weights["W{}".format(w_nb)], X) +
             weights["b{}".format(w_nb)] * np.ones((1, len(X[0]))))
        if w_nb < L:
            D = int(np.random.rand(A.shape[0], A.shape[1]) < keep_prob)
            A *= D
            A = np.tanh(Z)
            cache['D' + str(w_nb)] = D
        else:
            exp_Z = np.exp(Z)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        cache.update({"A{}".format(w_nb): A})
        X = cache["A{}".format(w_nb)]
    return (cache)
