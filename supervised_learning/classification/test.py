#!/usr/bin/env python3

import numpy as np
Deep = __import__('18-deep_neural_network').DeepNeuralNetwork

np.random.seed(18)
nx, m = np.random.randint(100, 1000, 2).tolist()
l = np.random.randint(3, 10)
sizes = np.random.randint(5, 20, l - 1).tolist()
sizes.append(1)
d = Deep(nx, sizes)
for i in range(l):
    d._DeepNeuralNetwork__weights['b' + str(i + 1)] = np.ones((sizes[i], 1))
X = np.random.randn(nx, m)
A, cache = d.forward_prop(X)
print(A)
for k, v in sorted(cache.items()):
    print(k, v)
