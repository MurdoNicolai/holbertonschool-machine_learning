#!/usr/bin/env python3

import numpy as np
Neuron = __import__('4-neuron').Neuron

np.random.seed(5)
nx, m = np.random.randint(100, 1000, 2).tolist()
nn = Neuron(nx)
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
Y_pred, C = nn.evaluate(X, Y)
print(Y_pred)
print(np.round(C, 10))

