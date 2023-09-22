#!/usr/bin/env python3
""" conv backward propagation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network
    """
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = ((h_new - 1) * sh + kh - A_prev.shape[1]) // 2
        pad_w = ((w_new - 1) * sw + kw - A_prev.shape[2]) // 2
        A_prev_padded = np.pad(A_prev,
                               ((0, 0), (pad_h, pad_h),
                                (pad_w, pad_w), (0, 0)),
                               mode='constant')
    else:
        A_prev_padded = A_prev

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    A_slice = A_prev_padded[i, vert_start:vert_end,
                                            horiz_start:horiz_end, :]
                    il, vl,  hl, = dA_prev[i, vert_start:vert_end,
                                           horiz_start:horiz_end, :].shape

                    i2l = A_slice.shape[0]

                    dA_prev[i, vert_start:vert_end,
                            horiz_start:horiz_end, :] += W[:il, :vl,
                                                           :hl, f] * dZ[i, h,
                                                                        w, f]
                    dW[:i2l, :, :, f] += A_slice * dZ[i, h, w, f]

    return dA_prev, dW, db
