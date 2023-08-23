#!/usr/bin/env python3

import numpy as np
Neuron = __import__('2-neuron').Neuron

np.random.seed(3)
nx, m = np.random.randint(100, 1000, 2).tolist()
nn = Neuron(nx)
nn._Neuron__b = 1
X = np.random.randn(nx, m)
A = nn.forward_prop(X)
print(A)


