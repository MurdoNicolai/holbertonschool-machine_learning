#!/usr/bin/env python3
"""containst gradient_descent functino"""
import numpy as np
import tensorflow.compat.v1 as tf


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates the weights and biases of a neural network
    using gradient descent with L2 regularization"""
    weights_sum = 0
    for key, value in weights.items():
        if key[:1] == 'W':
            weights_sum += np.sum(np.multiply(value, value))
    return (weights_sum/(2*m) * lambtha + cost)
