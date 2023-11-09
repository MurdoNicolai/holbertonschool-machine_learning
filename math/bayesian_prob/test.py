#!/usr/bin/env python3

intersection = __import__('1-intersection').intersection
import numpy as np

try:
    intersection(20, 25, np.linspace(0, 1, 5), [0.2, 0.2, 0.2, 0.2, 0.2])
except TypeError as e:
    print(str(e))
try:
    intersection(20, 25, np.linspace(0, 1, 5), np.array([0.5, 0.5]))
except TypeError as e:
    print(str(e))
try:
    intersection(20, 25, np.linspace(0, 1, 5), np.array([[0.2, 0.2, 0.2, 0.2, 0.2]]))
except TypeError as e:
    print(str(e))
