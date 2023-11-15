#!/usr/bin/env python3

import numpy as np
variance = __import__('2-variance').variance

if __name__ == "__main__":
    X = np.random.randn(100, 3)
    print(variance(X, 'hello'))
    print(variance(X, np.array([1, 2, 3, 4, 5])))
    print(variance(X, np.array([[[1, 2, 3, 4, 5]]])))
    print(variance(X, np.random.randn(5, 6)))
