#!/usr/bin/env python3

import numpy as np
optimum_k = __import__('3-optimum').optimum_k

if __name__ == "__main__":
    np.random.seed(1)
    means = np.random.uniform(0, 100, (2, 6))
    a = np.random.multivariate_normal(means[0], 10 * np.eye(6), size=10)
    b = np.random.multivariate_normal(means[1], 10 * np.eye(6), size=10)
    X = np.concatenate((a, b), axis=0)
    np.random.shuffle(X)
    res, v = optimum_k(X)
    print(res)
    print(np.round(v, 5))
