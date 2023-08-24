#!/usr/bin/env python3

import numpy as np
NN = __import__('14-neural_network').NeuralNetwork

np.random.seed(14)
nx, l, m = np.random.randint(10, 100, 3).tolist()
nn = NN(nx, l)
X = np.random.randn(nx, m)
Y = np.random.randint(0, 2, (1, m))
A, cost = nn.train(X, Y)
print(A)
print(np.round(cost, decimals=10))
print(np.round(nn.W1, decimals=10))
print(np.round(nn.b1, decimals=10))
print(np.round(nn.A1, decimals=10))
print(np.round(nn.W2, decimals=10))
print(np.round(nn.b2, decimals=10))
print(np.round(nn.A2, decimals=10))
try:
    nn.A1 = 10
    print('Fail: Private attribute A1 overwritten as public attribute')
except:
    pass
try:
    nn.W1 = 10
    print('Fail: Private attribute W1 overwritten as public attribute')
except:
    pass
try:
    nn.b1 = 10
    print('Fail: Private attribute b1 overwritten as public attribute')
except:
    pass
try:
    nn.A2 = 10
    print('Fail: Private attribute A2 overwritten as public attribute')
except:
    pass
try:
    nn.W2 = 10
    print('Fail: Private attribute W2 overwritten as public attribute')
except:
    pass
try:
    nn.b2 = 10
    print('Fail: Private attribute b2 overwritten as public attribute')
except:
    pass
