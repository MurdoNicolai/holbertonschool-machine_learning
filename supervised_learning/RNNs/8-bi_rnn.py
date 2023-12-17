#!/usr/bin/env python3
"""contains classes for rnn"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Parameters:
    - bi_cell: Instance of BidirectionalCell used for the forward propagation.
    - X: Data to be used, numpy.ndarray of shape (t, m, i).
    - h_0: Initial hidden state in the forward direction
    - h_t: Initial hidden state in the backward direction


    Returns:
    - H: Numpy.ndarray containing all of the concatenated hidden states.
    - Y: Numpy.ndarray containing all of the outputs.
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t, m, 2 * h))  # Concatenated hidden states
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))  # Outputs

    h_prev_forward = h_0
    h_prev_backward = h_t

    for t_step in range(t):
        x_t = X[t_step]

        # Forward direction
        h_next_forward = bi_cell.forward(h_prev_forward, x_t)

        # Backward direction
        h_next_backward = bi_cell.backward(h_prev_backward, x_t)

        # Concatenate hidden states
        H[t_step] = np.concatenate((h_next_forward, h_next_backward), axis=1)

        # Output
        Y[t_step] = bi_cell.output(H[t_step])

        # Update previous hidden states
        h_prev_forward = h_next_forward
        h_prev_backward = h_next_backward

    return H, Y
