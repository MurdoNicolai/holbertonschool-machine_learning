#!/usr/bin/env python3
"""contains one_hot_encode function"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts numerical base vector into matrix"""
    one_hot_matrix = np.zeros((classes, len(Y)))
    for row in range(len(Y)):
        num = Y[row]
        if num >= classes:
            return None
        one_hot_matrix[num - classes][row] = 1
    return one_hot_matrix
