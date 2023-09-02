#!/usr/bin/env python3
""" moving average"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using the gradient descent
        with momentum optimization algorithm """
    v_new = ((beta1 * v) + ((1 - beta1) * grad))
    var_new = var - v_new * alpha
    return var_new, v_new
