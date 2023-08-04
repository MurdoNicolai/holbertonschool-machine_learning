#!/usr/bin/env python3
"""poly_derivative function"""


def poly_derivative(poly):
    """Compute the derivative of a polynomial distribution"""
    new_poly = []
    position = 0
    for value in poly:
        if type(value) is not int or type(value) is not float:
            return None
        new_poly.append(value * position)
        position += 1
    return new_poly
