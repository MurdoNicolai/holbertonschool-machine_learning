#!/usr/bin/env python3

import numpy as np
from multinormal import MultiNormal

np.random.seed(3)
X = np.random.multivariate_normal([5], [[6]], 10000).T
mn = MultiNormal(X)
print(mn.mean)
print(mn.cov)
