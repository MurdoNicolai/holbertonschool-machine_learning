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
        Z1 = np.matmul(self.W1, X) + self.b1 * np.ones((1, len(X[0])))
        self.__A1 = 1.0 / (1.0 + np.exp(-1 * Z1))
        self.__A2 = 1.0 / (1.0 + np.exp(-1 * (np.matmul(self.W2, self.__A1)
                           + self.b2 * np.ones((1, len(self.__A1[0]))))))
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """return the cost of the neuron"""
        return(np.average(-np.log(abs((1.0000001 - Y*1.0000001) - A))))

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        label = self.forward_prop(X)
        prediction = self.cost(Y, label[1])
        return (label[1].round().astype(int), prediction)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """creates the training operation for the network"""
        dZ2 = A2 - Y
        dW2 = np.mean((dZ2 * A1), axis=1)
        db2 = np.average(dZ2, axis=1)

        dg = A1 * (1 - A1)
        dZ1 = (np.matmul(self.__W2.T, dZ2)) * dg
        dW1 = np.matmul(dZ1, X.T)/len(X[0])
        db1 = np.resize(np.sum(dZ1, axis=1), (len(A1), 1))/(len(X[0]))
        self.__W2[0] = self.__W2[0] - alpha * dW2
        self.__b2 = np.resize((self.__b2 - alpha * db2), (1, 1))
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=False, graph=False, step=100):
        """Trains the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        elif iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        elif alpha < 0:
            raise ValueError("alpha must be positive")

        grapheY = []
        for iterati in range(iterations):

            A1, A2 = self.forward_prop(X)

            self.gradient_descent(X, Y, A1, A2, alpha)

            # if verbose or graph:
            #     if type(step) is not int:
            #         raise TypeError("step must be an integer")
            #     elif step < 1 or step > iterations:
            #       raise ValueError("step must be positive and <= iterations")
            #    if verbose and (iterati % step == 0 or iterati == iterations):
            #         print(
            #           f"Cost after {iterati} iterations: {self.cost(Y, A1)}")
            #     if graph and (iterati % step == 0 or iterati == iterations):
            #         grapheY.append(self.cost(Y, A1))

        # if graph:
        #     x = np.linspace(0, iterations, iterations/step)
        #     y = grapheY
        #     plt.plot(x, y)
        #     plt.xlabel("iteration")
        #     plt.ylabel("cost")
        #     plt.title("Training Cost")
        #     plt.show()
        return self.evaluate(X, Y)
