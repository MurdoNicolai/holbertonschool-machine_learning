#!/usr/bin/env python3
def poly_integral(poly, C=0):
    """Returns the integral of the given polynomial"""
    new_poly = []
    position = len(poly) + 1
    for value in poly:
        if type(value) is not int or type(value) is not float:
            return None
        new_poly.append(value / position)
        position -= 1
    return new_poly.append(C)
