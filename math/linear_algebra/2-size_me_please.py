#!/usr/bin/env python3


def matrix_shape(matrix):
    elem = matrix
    size = []
    while type(elem) is list:
        nb_elem = len(elem)
        size.append(nb_elem)
        elem = elem[0]
    return size
