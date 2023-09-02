#!/usr/bin/env python3

import numpy as np

update_variables_momentum = __import__('5-momentum').update_variables_momentum


np.random.seed(5)
a, b = np.random.uniform(low=0.01, size=2)
m, nv = np.random.randint(10, 100, 2)
v = np.random.randn(m, nv)
dv = np.random.randn(m, nv)
dv_p = np.random.randn(m, nv)
print(update_variables_momentum(a, b, v, dv, dv_p))
