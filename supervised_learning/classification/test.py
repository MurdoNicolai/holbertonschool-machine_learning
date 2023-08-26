#!/usr/bin/env python3

import numpy as np
Deep = __import__('26-deep_neural_network').DeepNeuralNetwork

deep = Deep.load('26-test.pkl')
np.set_printoptions(threshold=np.inf)
print(deep.L)
for k, v in sorted(deep.cache.items()):
    print(k, v)
for k, v in sorted(deep.weights.items()):
    print(k, v)
