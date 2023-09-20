#!/usr/bin/env python3
""" conv_forward"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional
       layer of a neural network:"""
    filter_height, filter_width = W.shape[0], W.shape[1]
    if padding == "same":
        padding_height = ((len(A_prev[0]) - 1) * stride[0] -
                          len(A_prev[0]) + filter_height)
        padding_width = ((len(A_prev[0][0]) - 1) * stride[1] -
                         len(A_prev[0][0]) + filter_width)
        ph1 = int(padding_height/2)
        ph2 = int(padding_height - int(padding_height/2))
        pw1 = int(padding_width/2)
        pw2 = int(padding_width - int(padding_width/2))
        X = np.pad(A_prev,
                   [(0, 0), (ph1, ph2), (pw1, pw2), (0, 0)],
                   mode='constant')
    elif padding == "valid":
        padding_height = (len(A_prev[0]) - filter_height) % stride[0]
        padding_width = (len(A_prev[0][0]) - filter_width) % stride[1]
        X = A_prev[:, 0:len(A_prev[0])-padding_height,
                   0:len(A_prev[0][0])-padding_width, :]
    else:
        print("padding musit be valid or same")

    bgW = np.expand_dims(W, axis=0)
    bgW = np.repeat(bgW, len(X), axis=0)
    bgb = np.expand_dims(b, axis=0)
    bgb = np.repeat(bgb, len(X), axis=0)

    bgb = bgb.reshape(len(X), len(b[0][0][0]))
    result = np.zeros((len(X), ((len(X[0]) - filter_height) // stride[0]) + 1,
                       ((len(X[0][0]) - filter_width) // stride[1]) + 1,
                       len(W[0][0][0])))

    for pos_h in range(0, len(X[0]) + 1 - filter_height, stride[0]):
        for pos_w in range(0, len(X[0][0]) + 1 - filter_width, stride[1]):
            extrait = X[:, pos_h:(pos_h + filter_height),
                        pos_w:(pos_w + filter_width), :]
            bgextrait = np.expand_dims(extrait, axis=-1)
            bgextrait = np.repeat(bgextrait, len(W[0][0][0]), axis=-1)

            sum = np.sum(np.multiply(bgW, bgextrait), axis=(1, 2, 3)) + bgb
            result[:, pos_h // stride[0],
                   pos_w // stride[1], :] = activation(sum)

    return result
