#!/usr/bin/env python3
"""contains add function"""


def add_matrices(mat1, mat2):
    """returns sum of 2 matrices"""
    if ((type(mat1) is int or type(mat1) is float)
       and (type(mat2) is int or type(mat2) is float)):
        return mat1 + mat2

    elif isinstance(mat1, list) or isinstance(mat1, list):
        return None

    if len(mat1) != len(mat2):
        return None

    result = []
    for row_nb in range(len(mat1)):
        if add_matrices(mat1[row_nb], mat2[row_nb]):
            result.append(add_matrices(mat1[row_nb], mat2[row_nb]))
        else:
            return None

    return result
