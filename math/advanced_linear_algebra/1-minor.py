#!/usr/bin/env python3
"""contains matrix math functions"""


def determinant(matrix):
    """ calculates the determinant of a matrix """
    if matrix == [[]]:
        return 1
    if (type(matrix) != list):
        raise TypeError('matrix must be a list of lists')
    else:
        for row in matrix:
            if type(row) != list:
                raise TypeError('matrix must be a list of lists')

    if len(matrix[0]) != len(matrix):
        raise ValueError('matrix must be a square matrix')

    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1]
    if len(matrix) == 1:
        return matrix[0][0]

    det = 0
    for i in range(len(matrix[0])):
        temp_matrix = matrix.copy()
        del temp_matrix[0]
        for row in range(len(temp_matrix)):
            temp_matrix[row] = temp_matrix[row].copy()
        for row in temp_matrix:
            del row[i]
        det += (matrix[0][i] * (-1)**i) * determinant(temp_matrix)
    return det


def minor(matrix):
    """ calculates the minor of a matrix """
    if matrix == [[]]:
        return [[1]]
    if (type(matrix) != list):
        raise TypeError('matrix must be a list of lists')
    else:
        for row in matrix:
            if type(row) != list:
                raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1:
        return [[1]]

    new_matrix = [0] * len(matrix)
    for row in range(len(new_matrix)):
        new_matrix[row] = [0] * len(matrix[0])

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            temp_matrix = matrix.copy()
            del temp_matrix[i]
            for row in range(len(temp_matrix)):
                temp_matrix[row] = temp_matrix[row].copy()
            for row in temp_matrix:
                del row[j]
            new_matrix[i][j] = determinant(temp_matrix)
    return new_matrix
