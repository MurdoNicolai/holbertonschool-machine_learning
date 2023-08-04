#!/usr/bin/env python3
"""poly_integral"""


def poly_integral(poly, C=0):
    """Returns the integral of the given polynomial"""
    new_poly = []
    position =  1
    for value in poly:
        if type(value) is not int and type(value) is not float:
            return type(value)
        result = value / position
        if result%1 == 0:
            result = int(result)
        new_poly.append(result)
        position += 1
    new_poly.append(C)
    return new_poly
