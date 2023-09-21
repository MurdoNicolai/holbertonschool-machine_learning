#!/usr/bin/env python3
""" conv_forward"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a convolutional
       layer of a neural network:"""
    filter_height, filter_width = kernel_shape[0], kernel_shape[1]
    X = A_prev
    result = np.zeros((len(X), ((len(X[0]) - filter_height) // stride[0]) + 1,
                       ((len(X[0][0]) - filter_width) // stride[1]) + 1,
                       len(X[0][0][0])) )
    for pos_h in range(0, len(X[0]) + 1 - filter_height, stride[0]):
        for pos_w in range(0, len(X[0][0]) + 1 - filter_width, stride[1]):
            extrait = X[:, pos_h:(pos_h + filter_height),
                        pos_w:(pos_w + filter_width), :]
            if mode == 'max':
                sum = np.max(extrait, axis=(1, 2))
            else:
                sum = np.avg(extrait, axis=(1, 2))
            result[:, pos_h // stride[0],
                   pos_w // stride[1], :] = sum

    return result
