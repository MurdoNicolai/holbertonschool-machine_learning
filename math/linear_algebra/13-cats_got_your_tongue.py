#!/usr/bin/env python3
"""contains calc function"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """returns tuple with add sub mul ad div of both matrices"""
    return np.concatenate((mat1, mat2), axis=axis)
