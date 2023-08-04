#!/usr/bin/env python3
"""poly_derivative function"""


def poly_derivative(poly):
    """Compute the derivative of a polynomial distribution"""
    new_poly = []
    position = 1
    if type(poly) is not list:
        return None
    if len(poly) == 0:
        return None
    for value in poly[1:]:
        if type(value) is not int and type(value) is not float:
            return None
        new_poly.append(value * position)
        position += 1
    if new_poly == []:new_poly = [0]
    return new_poly
