#!/usr/bin/env python3
"""contains one_hot_encode function"""
import numpy as np


def one_hot_decode(one_hot):
    """converts one_hot matrix into numerical base vector"""
    if ((type(one_hot) is not np.ndarray
       or type(one_hot[0]) is not np.ndarray
       or type(one_hot[0][0]) is not (np.float64 or np.int))):
        return None
    nb_in_row = len(one_hot)
    numbers = np.array(range(0, nb_in_row))
    return np.matmul(numbers, one_hot.astype(int))
