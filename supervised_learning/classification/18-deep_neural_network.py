#!/usr/bin/env python3
"""contains neuron class"""
import numpy as np


class DeepNeuralNetwork():
    """ deffines the neuron for a network"""
    def __init__(self, nx, layers):
        """
        initialize the neuronL:
        L: The number of layers in the neural network.
        cache: A dictionary to hold all intermediary values of the network
        weights: A dictionary to hold all weights and biased of the network.

        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        else:
            self.__weights = {}
            for layer in range(len(layers)):
                nodes = layers[layer]
                if type(nodes) is not int or nodes < 0:
                    raise TypeError(
                            "layers must be a list of positive integers")
                else:
                    w_std = np.sqrt(2.0 / nx)
                    self.__weights.update({"W{}".format(layer + 1):
                                           np.random.randn(nodes, nx) * w_std})
                    self.__weights.update({"b{}".format(layer + 1):
                                           np.zeros((nodes, 1))})
                    nx = nodes
            self.__L = len(layers)
            self.__cache = {}

    @property
    def L(self):
        """I'm the 'L' property."""
        return self.__L

    @property
    def cache(self):
        """I'm the 'cache' property."""
        return self.__cache

    @property
    def weights(self):
        """I'm the 'weights' property."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neurons"""
        self.__cache.update({"A0": X})
        w_nb = 0
        while w_nb < len(self.weights)/2:
            w_nb = w_nb + 1
            Z = (np.matmul(self.weights["W{}".format(w_nb)], X) +
                 self.weights["b{}".format(w_nb)] * np.ones((1, len(X[0]))))
            self.__cache.update({"A{}".format(w_nb):
                                 1.0 / (1.0 + np.exp(-1 * Z))})
            X = self.cache["A{}".format(w_nb)]
        return (X, self.cache)

    # def cost(self, Y, A):
    #     """return the cost of the neuron"""
    #     return(np.average(-np.log(abs((1.0000001 - Y*1.0000001) - A))))

    # def evaluate(self, X, Y):
    #     """Evaluates the neuron’s predictions"""
    #     label = self.forward_prop(X)
    #     prediction = self.cost(Y, label[1])
    #     return (label[1].round().astype(int), prediction)

    # def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
    #     """creates the training operation for the network"""
    #     dZ2 = A2 - Y
    #     dW2 = np.mean((dZ2 * A1), axis=1)
    #     db2 = np.average(dZ2, axis=1)

    #     dg = A1 * (1 - A1)
    #     dZ1 = (np.matmul(self.__W2.T, dZ2)) * dg
    #     dW1 = np.matmul(dZ1, X.T)/len(X[0])
    #     db1 = np.resize(np.sum(dZ1, axis=1), (len(A1), 1))/(len(X[0]))
    #     self.__W2[0] = self.__W2[0] - alpha * dW2
    #     self.__b2 = np.resize((self.__b2 - alpha * db2), (1, 1))
    #     self.__W1 = self.__W1 - alpha * dW1
    #     self.__b1 = self.__b1 - alpha * db1

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

    #         A1, A2 = self.forward_prop(X)

    #         self.gradient_descent(X, Y, A1, A2, alpha)

    #         if verbose or graph:
    #             if type(step) is not int:
    #                 raise TypeError("step must be an integer")
    #             elif step < 1 or step > iterations:
    #               raise ValueError("step must be positive and <= iterations")
    #            if verbose and (iterati % step == 0 or iterati == iterations):
    #                 print(
    #                   f"Cost after {iterati} iterations: {self.cost(Y, A1)}")
    #             if graph and (iterati % step == 0 or iterati == iterations):
    #                 grapheY.append(self.cost(Y, A1))

    #     if graph:
    #         x = np.linspace(0, iterations, iterations/step)
    #         y = grapheY
    #         plt.plot(x, y)
    #         plt.xlabel("iteration")
    #         plt.ylabel("cost")
    #         plt.title("Training Cost")
    #         plt.show()
    #     return self.evaluate(X, Y)
