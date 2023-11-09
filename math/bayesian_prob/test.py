#!/usr/bin/env python3

posterior = __import__('3-posterior').posterior
import numpy as np

try:
    posterior('1', 50, np.linspace(0, 1, 11), np.ones(11) / 11)
except ValueError as e:
    print(str(e))
try:
    posterior(-1, 50, np.linspace(0, 1, 11), np.ones(11) / 11)
except ValueError as e:
    print(str(e))
posterior(0, 50, np.linspace(0, 1, 11), np.ones(11) / 11)
