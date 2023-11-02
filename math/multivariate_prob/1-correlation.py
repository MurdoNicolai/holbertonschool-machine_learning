#!/usr/bin/env python3
"""contains a mean calculator"""
import numpy as np


import numpy as np

def correlation(C):
    """Returns the correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    d, d2 = C.shape
    if d != d2:
        raise ValueError("C must be a 2D square matrix")

    std_dev = np.sqrt(np.diag(C))
    correlation_matrix = C / np.outer(std_dev, std_dev)

    return correlation_matrix
