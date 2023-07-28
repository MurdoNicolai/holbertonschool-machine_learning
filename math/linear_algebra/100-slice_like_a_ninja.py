#!/usr/bin/env python3
"""contains slice function"""


def np_slice(matrix, axes={}):
    """returns slice a matrix along a axis, axes is a dictionary where the key
    is an axis to slice along and the value is a tuple representing the slice
    to make along that axis"""

    key = 0
    start = stop = step = None
    if key in axes:
        if len(axes[key]) > 0:
            start = axes[key][0]
        if len(axes[key]) > 1:
            stop = axes[key][1]
        else:
            stop = axes[key][0]
            start = 0
        if len(axes[key]) > 2:
            step = axes[key][2]
    d0 = slice(start, stop, step)

    key = 1
    start = stop = step = None
    if key in axes:
        if len(axes[key]) > 0:
            start = axes[key][0]
        if len(axes[key]) > 1:
            stop = axes[key][1]
        else:
            stop = axes[key][0]
            start = 0
        if len(axes[key]) > 2:
            step = axes[key][2]
    d1 = slice(start, stop, step)

    key = 2
    start = stop = step = None
    if key in axes:
        if len(axes[key]) > 0:
            start = axes[key][0]
        if len(axes[key]) > 1:
            stop = axes[key][1]
        else:
            stop = axes[key][0]
            start = 0
        if len(axes[key]) > 2:
            step = axes[key][2]
    d2 = slice(start, stop, step)

    key = 3
    start = stop = step = None
    if key in axes:
        if len(axes[key]) > 0:
            start = axes[key][0]
        if len(axes[key]) > 1:
            stop = axes[key][1]
        else:
            stop = axes[key][0]
            start = 0
        if len(axes[key]) > 2:
            step = axes[key][2]
    d3 = slice(start, stop, step)

    key = 4
    start = stop = step = None
    if key in axes:
        if len(axes[key]) > 0:
            start = axes[key][0]
        if len(axes[key]) > 1:
            stop = axes[key][1]
        else:
            stop = axes[key][0]
            start = 0
        if len(axes[key]) > 2:
            step = axes[key][2]
    d4 = slice(start, stop, step)

    key = 5
    start = stop = step = None
    if key in axes:
        if len(axes[key]) > 0:
            start = axes[key][0]
        if len(axes[key]) > 1:
            stop = axes[key][1]
        else:
            stop = axes[key][0]
            start = 0
        if len(axes[key]) > 2:
            step = axes[key][2]
    d5 = slice(start, stop, step)

    if matrix.ndim == 1:
        matrix = matrix[d0]
    if matrix.ndim == 2:
        matrix = matrix[d0, d1]
    if matrix.ndim == 3:
        matrix = matrix[d0, d1, d2]
    if matrix.ndim == 4:
        matrix = matrix[d0, d1, d2, d3]
    if matrix.ndim == 5:
        matrix = matrix[d0, d1, d2, d3, d4]
    if matrix.ndim == 6:
        matrix = matrix[d0, d1, d2, d3, d4, d5]

    return matrix
