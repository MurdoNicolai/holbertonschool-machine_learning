#!/usr/bin/env python3
def poly_derivative(poly):
    """Compute the derivative of a polynomial distribution"""
    new_poly = []
    position = len(poly)
    for value in poly:
        if type(value) is not int or type(value) is not float:
            return None
        new_poly.append(value * position)
        position -= 1
    return new_poly
