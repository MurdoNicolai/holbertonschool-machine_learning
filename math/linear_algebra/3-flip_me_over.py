#!/usr/bin/env python3


def matrix_transpose(matrix):
    # return matrix transposed
    rowLength = len(matrix[0])
    trans_mat = []
    for rows in range(rowLength):
        trans_mat.append([])

    for row in matrix:
        for position in range(rowLength):
            trans_mat[position].append(row[position])

    return trans_mat
