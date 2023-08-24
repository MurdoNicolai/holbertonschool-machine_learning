#!/usr/bin/env python3
"""contains neuron class"""
import numpy as np

class NeuralNetwork():
    """ deffines the neuron for a network"""
    def __init__(self, nx, nodes):
        """
        initialize the neuron

        W1: The weights vector for the hidden layer.
        b1: The bias for the hidden layer.
        A1: The activated output for the hidden layer.
        W2: The weights vector for the output neuron.
        b2: The bias for the output neuron.
        A2: The activated output for the output neuron (prediction).

        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        else:
            self.W1 = np.random.randn(nodes, nx)
            self.b1 = np.zeros((nodes, 1), dtype=float)
            self.A1 = 0
            self.W2 = np.random.randn(1, nodes)
            self.b2 = 0
            self.A2 = 0
