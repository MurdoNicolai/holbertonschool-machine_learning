#!/usr/bin/env python3
import numpy as np

def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, _ = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = ((h_new - 1) * sh + kh - A_prev.shape[1]) // 2
        pad_w = ((w_new - 1) * sw + kw - A_prev.shape[2]) // 2
        A_prev_padded = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
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

                    if padding == "same":
                        A_slice = A_prev_padded[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    else:
                        A_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, f] * dZ[i, h, w, f]
                    dW[:, :, :, f] += A_slice * dZ[i, h, w, f]

    if padding == "valid":
        dA_prev = dA_prev[:, kh - 1:-(kh - 1), kw - 1:-(kw - 1), :]

    return dA_prev, dW, db
