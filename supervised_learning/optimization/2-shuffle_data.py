#!/usr/bin/env python3
""" contains normalization"""
import numpy as np


def shuffle_data(X, Y):
    """" that shuffles the data points in two matrices the same way"""
    print(X[0], Y[0])
    lenX = X[0].size
    fuse = np.concatenate((X, Y), axis=1)
    shuffle = np.random.permutation(fuse)
    split = np.split(shuffle, shuffle[0].size, axis=1)
    X = np.concatenate(split[:lenX], axis=1)
    Y = np.concatenate(split[lenX:], axis=1)
    return(X, Y)
