#!/usr/bin/env python3

"""contains multiply function"""


def mat_mul(mat1, mat2):
    """2 matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None

    result_matrix = []
    for row_nb in range(len(mat1)):
        result_matrix_row = []
        for col_nb in range(len(mat2[0])):
            prod = 0
            for position in range(len(mat1[0])):
                prod += mat1[row_nb][position] * mat2[position][col_nb]
            result_matrix_row.append(prod)
        result_matrix.append(result_matrix_row)

    return result_matrix
