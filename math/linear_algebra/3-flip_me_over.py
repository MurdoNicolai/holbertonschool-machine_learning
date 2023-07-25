#!/usr/bin/env python3
import numpy


def matrix_transpose(matrix):
    # return matrix transposed
    rowLength = len(matrix[0])
    trans_mat = numpy.ones((rowLength, len(matrix)))
    nbrow = 0
    for row in matrix:
        for position in range(rowLength):
            trans_mat[position][nbrow] = row[position]
        nbrow += 1
    return trans_mat.astype(int)
