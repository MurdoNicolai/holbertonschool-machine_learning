#!/usr/bin/env python3
"""summation_i_squared function"""


def summation_i_squared(n):
    """Returns the summation"""
    if n < 1:
        return None
    if n == 1:
        return 1
    # https://en.wikipedia.org/wiki/List_of_mathematical_series#Sums_of_powers
    return n * n * n / 3 + n * n / 2 + n / 6
