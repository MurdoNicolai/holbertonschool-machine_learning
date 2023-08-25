#!/usr/bin/env python3

import numpy as np
Deep = __import__('21-deep_neural_network').DeepNeuralNetwork

np.random.seed(21)
nx, m = np.random.randint(100, 1000, 2).tolist()
l = np.random.randint(3, 10)
sizes = np.random.randint(5, 20, l - 1).tolist()
sizes.append(1)
d = Deep(nx, sizes)
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
_, cache = d.forward_prop(X)

for k, v in sorted(d.weights.items()):
    print(k, v)
d.gradient_descent(Y, cache, alpha=0.5)
for k, v in sorted(d.weights.items()):
    print(k, v)
try:
    d.weights = 10
    print('Fail: private attribute weights is overwritten as a public attribute')
except:
    pass
