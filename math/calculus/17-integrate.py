#!/usr/bin/env python3
"""poly_integral"""


def poly_integral(poly, C=0):
    """Returns the integral of the given polynomial"""
    new_poly = []
    position =  1
    for value in poly:
        if type(value) is not int and type(value) is not float:
            return type(value)
        new_poly.append(value / position)
        position += 1
    new_poly.append(C)
    return new_poly

print(poly_integral([7, 4, 6, 1, 5]))
print(poly_integral([4, 8, 2, 4, 7, 1, 9], C=5))

