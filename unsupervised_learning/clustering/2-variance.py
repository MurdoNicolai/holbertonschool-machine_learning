
#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set"""
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)
    var = np.sum((X - C[clss]) ** 2)
    return (var)
