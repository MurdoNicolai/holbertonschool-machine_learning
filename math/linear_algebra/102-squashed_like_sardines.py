#!/usr/bin/env python3
"""contains calc function"""


def cat_matrices(mat1, mat2, axis=0):
    """concatenate matrices along the axis specified"""
    result = []
    if axis == 0:
        for row_nb in range(len(mat1)):
            result.append(mat1[row_nb])
        for row_nb in range(len(mat2)):
            result.append(mat2[row_nb])
        return result
    elif len(mat1) != len(mat2):
        return None
    else:
        for row_nb in range(len(mat1)):
            if cat_matrices(mat1[row_nb], mat2[row_nb], axis - 1) is None:
                return None
            result.append(cat_matrices(mat1[row_nb], mat2[row_nb], axis - 1))
        return(result)
