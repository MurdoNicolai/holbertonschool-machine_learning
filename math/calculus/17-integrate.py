#!/usr/bin/env python3
"""poly_integral"""


def poly_integral(poly, C=0):
    """Returns the integral of the given polynomial"""
    new_poly = [C]
    position = 1
    if type(poly) is not list:
        return None
    if len(poly) == 0 or type(C) is not int:
        return None
    for value in poly:
        if type(value) is not int and type(value) is not float:
            return type(value)
        result = value / position
        if result % 1 == 0:
            result = int(result)
        new_poly.append(result)
        position += 1
    for pos in range(len(new_poly)-1, 0, -1):
        if new_poly[pos] == 0:
            new_poly.pop()
        else:
            break
    return new_poly
