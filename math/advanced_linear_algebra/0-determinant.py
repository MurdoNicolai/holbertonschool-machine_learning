#!/usr/bin/env python3
"""contains matrix math functions"""
import numpy as np


def determinant(matrix):
    """ calculates the determinant of a matrix """
    if matrix == [[]]:
        return 0
    try:
        matrix = np.array(matrix)
        if matrix.ndim != 2:
            raise TypeError('matrix must be a list of lists')
    except TypeError:
        raise TypeError('matrix must be a list of lists')
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('matrix must be a square matrix')

    if matrix.shape[0] == 2:
        return matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1]
    if matrix.shape[0] == 1:
        return matrix[0][0]

    det = 0
    for i in range(len(matrix[0])):
        temp_matrix = matrix
        temp_matrix = np.delete(temp_matrix, 0, 0)
        temp_matrix = np.delete(temp_matrix, i, 1)
        det += (matrix[0][i] * (-1)**i) * determinant(temp_matrix)
    return det
