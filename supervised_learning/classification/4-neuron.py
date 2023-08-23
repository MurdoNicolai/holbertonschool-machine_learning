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
            self.__W = np.random.randn(1, nx)
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """I'm the 'W' property."""
        return self.__W

    @property
    def b(self):
        """I'm the 'b' property."""
        return self.__b

    @property
    def A(self):
        """I'm the 'A' property."""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        temp = 1.0 / (1.0 + np.exp(-1 * (np.matmul(self.W, (X))
                                         + (np.ones(X.shape) * self.b))))
        self.__A = (np.ones((1, np.size(temp[0]))))
        self.__A[0] = temp[0]
        return self.__A

    def cost(self, Y, A):
        """return the cost of the neuron"""
        return(np.average(-np.log(abs((1.0000001 - Y*1.0000001) - A))))

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        label = self.forward_prop(X)
        prediction = self.cost(Y, label)
        return (label.round(decimals=0), prediction)
