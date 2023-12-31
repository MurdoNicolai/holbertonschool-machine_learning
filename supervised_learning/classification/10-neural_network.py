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
            self.__W1 = np.random.randn(nodes, nx)
            self.__b1 = np.zeros((nodes, 1), dtype=float)
            self.__A1 = 0
            self.__W2 = np.random.randn(1, nodes)
            self.__b2 = 0
            self.__A2 = 0

    @property
    def W1(self):
        """I'm the 'Weight' property."""
        return self.__W1

    @property
    def b1(self):
        """I'm the 'bias' property."""
        return self.__b1

    @property
    def A1(self):
        """I'm the 'Activation' property."""
        return self.__A1

    @property
    def W2(self):
        """I'm the 'Weight' property."""
        return self.__W2

    @property
    def b2(self):
        """I'm the 'bias' property."""
        return self.__b2

    @property
    def A2(self):
        """I'm the 'Activation' property."""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        self.__A1 = 1.0 / (1.0 + np.exp(-1 * (np.matmul(self.W1, X)
                           + np.matmul(self.b1, np.ones((1, len(X[0])))))))
        self.__A2 = 1.0 / (1.0 + np.exp(-1 * (np.matmul(self.W2, self.__A1)
                           + self.b2 * np.ones((1, len(self.__A1[0]))))))
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """return the cost of the neuron"""
        return(np.average(-np.log(abs((1.0000001 - Y*1.0000001) - A))))

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        label = self.forward_prop(X)
        prediction = self.cost(Y, label)
        return (label.round().astype(int), prediction)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """creates the training operation for the network"""
        dZ = A - Y
        self.__b = self.__b - alpha * np.average(dZ)
        self.__W = self.__W - np.mean((alpha * X * dZ), axis=1)

    # def train(self, X, Y, iterations=5000, alpha=0.05,
    #           verbose=True, graph=True, step=100):
    #     """Trains the neuron"""
    #     if type(iterations) is not int:
    #         raise TypeError("iterations must be an integer")
    #     elif iterations < 1:
    #         raise ValueError("iterations must be a positive integer")
    #     if type(alpha) is not float:
    #         raise TypeError("alpha must be a float")
    #     elif alpha < 0:
    #         raise ValueError("alpha must be positive")

    #     grapheY = []
    #     foriterati in range(iterations):

    #         A = self.forward_prop(X)

    #         self.gradient_descent(X, Y, A, alpha)

    #         if verbose or graph:
    #             if type(step) is not int:
    #                 raise TypeError("step must be an integer")
    #             elif step < 1 or step > iterations:
    #               raise ValueError("step must be positive and <= iterations")
    #            if verbose and (iterati % step == 0 or iterati == iterations):
    #                 print(
    #                   f"Cost after {iterati} iterations: {self.cost(Y, A)}")
    #             if graph and (iterati % step == 0 or iterati == iterations):
    #                 grapheY.append(self.cost(Y, A))

    #     if graph:
    #         x = np.linspace(0, iterations, iterations/step)
    #         y = grapheY
    #         plt.plot(x, y)
    #         plt.xlabel("iteration")
    #         plt.ylabel("cost")
    #         plt.title("Training Cost")
    #         plt.show()
    #     return self.evaluate(X, Y)
