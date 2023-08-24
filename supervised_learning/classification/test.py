#!/usr/bin/env python3

import numpy as np
NN = __import__('8-neural_network').NeuralNetwork

np.random.seed(8)
nx, l = np.random.randint(100, 1000, 2).tolist()
nn = NN(nx, l)
print(nn.W1)
print(nn.b1)
print(nn.A1)
print(nn.W2)
print(nn.b2)
print(nn.A2)
