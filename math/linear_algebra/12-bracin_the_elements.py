#!/usr/bin/env python3
"""contains calc function"""


def np_elementwise(mat1, mat2):
    """returns tuple with add sub mul ad div of both matrices"""
    return (mat1 + mat2, mat1 - mat2, mat2 * mat1, mat1 / mat2)
