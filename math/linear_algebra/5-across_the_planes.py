#!/usr/bin/env python3

"""contains add function"""


def add_matrices2D(mat1, mat2):
    """add matrices2D elem by elem"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    newmat = []
    for x in range(len(mat1)):
        newarray = []
        for y in range(len(mat1[0])):
            newarray.append(mat1[x][y] + mat2[x][y])
        newmat.append(newarray)
    return newmat
