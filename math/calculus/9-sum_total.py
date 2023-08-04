#!/usr/bin/env python3
"""summation_i_squared function"""


def summation_i_squared(n):
    """Returns the summation"""
    if n < 1:
        return None
    if n == 1:
        return 1
    a = summation_i_squared(n - 1)
    a = a + n * n
    return a
