#!/usr/bin/env python3
"""contains neuron class"""
import numpy as np


class Neuron():
    """ deffines the neuron for a network"""
    def __init__(self, nx):
        """
        initialize the neuron
        W: The weights vector for the neuron.
        b: The bias for the neuron
        A: The activated output of the neuron (prediction).
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.W = np.random.randn(1, nx)
            self.b = 0
            self.A = 0
