#!/usr/bin/env python3
""" contains normalization"""
import numpy as np


def shuffle_data(X, Y):
    """" that shuffles the data points in two matrices the same way"""
    fuse = np.concatenate((X, Y), axis=1)
    shuffle = np.random.permutation(fuse)
    X, Y = np.split(shuffle, 2, axis=1)
    return(X, Y)
