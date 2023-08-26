#!/usr/bin/env python3
"""contains neuron class"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

    def cost(self, Y, A):
        """return the cost of the neuron"""
        return(np.average(-np.log(abs((1.0000001 - Y*1.0000001) - A))))

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        label = self.forward_prop(X)
        prediction = self.cost(Y, label[0])
        return (label[0].round().astype(int), prediction)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """creates the training operation for the network"""
        A = cache["A{}".format(len(cache) - 1)]
        da = -(Y/A)+((1-Y)/(1-A))
        len_cache = len(cache)
        newweights = {}

        for i in range(len_cache - 1, 0, -1):
            A2 = cache["A{}".format(i)]
            A1 = cache["A{}".format(i - 1)]
            dg2 = A2 * (1 - A2)
            dZ = (da) * dg2
            dW = np.matmul(dZ, A1.T)/len(A1[0])
            db = np.resize(np.sum(dZ, axis=1), (len(A2), 1))/(len(A1[0]))
            da = np.matmul(self.weights["W{}".format(i)].T, dZ)
            newweights.update({"W{}".format(i):
                              self.weights["W{}".format(i)] - alpha * dW})
            newweights.update({"b{}".format(i):
                              self.weights["b{}".format(i)] - alpha * db})
        self.__weights = newweights

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

            A, cache = self.forward_prop(X)

            self.gradient_descent(Y, cache, alpha)

            if verbose or graph:
                if type(step) is not int:
                    raise TypeError("step must be an integer")
                elif step < 1 or step > iterations:
                  raise ValueError("step must be positive and <= iterations")
                if verbose and (iterati % step == 0 or iterati == iterations):
                    print(
                      f"Cost after {iterati} iterations: {self.cost(Y, A)}")
                if graph and (iterati % step == 0 or iterati == iterations):
                    grapheY.append(self.cost(Y, A))

        if graph:
            x = np.linspace(0, iterations, iterations/step)
            y = grapheY
            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """save as pickle format"""
        if filename[-4:] != ".pkl":
            filename = filename+".pkl"
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load(filename):
        """save as pickle format"""
        try:
            with open('filename.pickle', 'rb') as handle:
                b = pickle.load(handle)
        except:
            return None
