#!/usr/bin/env python3

"""contains concatenation function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates two matrices along a axis(0 y axis, 1 x axis)"""

    if len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 0:
        return mat1 + mat2
    if len(mat1) != len(mat2):
        return None
    newmat = []
    for rownb in range(len(mat1)):
        newmat.append(mat1[rownb] + mat2[rownb])
    return newmat
