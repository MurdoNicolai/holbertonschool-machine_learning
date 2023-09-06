#!/usr/bin/env python3
"""containst reg_cost functino"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates the cost of a neural network with L2 regularization"""
    weights_sum = 0
    for value in weights.values():
        weights_sum += np.sum(np.multiply(value, value))
    return (weights_sum/(2*m) * lambtha + cost)
