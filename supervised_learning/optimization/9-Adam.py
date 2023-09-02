#!/usr/bin/env python3
""" moving average"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update a variable using the Adam optimization algorithm.
    """
    t += 1  # Increment the time step
    v = beta1 * v + (1 - beta1) * grad  # Update the first moment
    s = beta2 * s + (1 - beta2) * (grad ** 2)  # Update the second moment

    v_corrected = v / (1 - beta1 ** t)  # Bias-corrected first moment estimate
    s_corrected = s / (1 - beta2 ** t)  # Bias-corrected second moment estimate

    var -= alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)  # Update the variable

    return var, v, s


