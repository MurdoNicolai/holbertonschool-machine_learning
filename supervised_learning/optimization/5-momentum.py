#!/usr/bin/env python3
""" moving average"""
import numpy as np

def moving_average(data, beta):
    """ calculates moving average"""
    Average = 0
    total_numbers = 0
    moving_average = np.zeros(data.shape)
    for number in data:
        total_numbers += 1
        Average = ((beta * Average) + ((1 - beta) * number))

        moving_average[total_numbers - 1] = (Average/(1 - beta ** total_numbers))

    return moving_average.astype(np.float64)


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using the gradient descent
        with momentum optimization algorithm """
    grad = moving_average(grad, beta1)
    v_new = ((beta1 * v) + ((1 - beta1) * grad))
    var_new = var + v_new * alpha
    np.set_printoptions(precision=8, suppress=True)
    return var_new, v_new
