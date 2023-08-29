#!/usr/bin/env python3
""" contains normalization"""
import numpy as np


def shuffle_data(X, Y):
    """" that shuffles the data points in two matrices the same way"""
    return(np.random.permutation(X), np.random.permutation(Y))
