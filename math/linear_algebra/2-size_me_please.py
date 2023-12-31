#!/usr/bin/env python3
"""return the shape of the matrix"""


def matrix_shape(matrix):
    """return the shape of the matrix"""
    elem = matrix
    size = []
    while type(elem) is list:
        nb_elem = len(elem)
        size.append(nb_elem)
        elem = elem[0]
    return size
