#!/usr/bin/env python3
"""contains classes for rnn"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Parameters:
    - rnn_cells: List of RNNCell instances of length l (number of layers).
    - X: Data to be used, numpy.ndarray of shape (t, m, i).
    - h_0: Initial hidden state, numpy.ndarray of shape (l, m, h).

    Returns:
    - H: Numpy.ndarray containing all of the hidden states.
    - Y: Numpy.ndarray containing all of the outputs.
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))  # Hidden states
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))  # Outputs

    H[0] = h_0

    for t_step in range(t):
        x_t = X[t_step]

        for layer in range(l):
            h_prev = H[t_step, layer]
            rnn_cell = rnn_cells[layer]

            # Perform forward propagation for one time step
            H[t_step + 1, layer], y = rnn_cell.forward(h_prev, x_t)

            # Save output for the last layer
            if layer == l - 1:
                Y[t_step] = y

    return H, Y
