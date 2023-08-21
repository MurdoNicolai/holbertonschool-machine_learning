#!/usr/bin/env python3
"""contains neuron class"""
import numpy as np


class Neuron():
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.W = np.random.randn(1, nx)
            self.b = 0
            self.A = 0
