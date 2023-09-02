#!/usr/bin/env python3
""" moving average"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update a variable using the RMSProp optimization algorithm.
    """
    new_s = beta2 * s + (1 - beta2) * grad**2
    var -= alpha * grad / (np.sqrt(new_s) + epsilon)

    return var, new_s
