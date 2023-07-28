#!/usr/bin/env python3
"""contains slice function"""


def np_slice(matrix, axes={}):
    """returns slice a matrix along a axis, axes is a dictionary where the key
    is an axis to slice along and the value is a tuple representing the slice
    to make along that axis"""


    if "matrix_depth" not in axes:
        axes["matrix_depth"] = matrix.ndim
    if axes["matrix_depth"] - matrix.ndim in axes:
        start = stop = step = None
        if len(axes[axes["matrix_depth"] - matrix.ndim]) > 0:
            start = axes[axes["matrix_depth"] - matrix.ndim][0]
        if len(axes[axes["matrix_depth"] - matrix.ndim]) > 1:
            stop = axes[axes["matrix_depth"] - matrix.ndim][1]
        if len(axes[axes["matrix_depth"] - matrix.ndim]) > 2:
            step = axes[axes["matrix_depth"] - matrix.ndim][2]
        matrix = matrix[slice(start, stop, step)]



    if type(matrix[0]) is not type(np.array([])):
        return matrix

    temp_matrix = []
    for row_nb in range(len(matrix)):
        temp_matrix.append(np_slice(matrix[row_nb], axes))

    matrix = np.array(temp_matrix)

    return matrix
