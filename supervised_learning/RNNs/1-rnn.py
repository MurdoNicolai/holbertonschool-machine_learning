#!/usr/bin/env python3
"""contains classes for rnn"""
import numpy as np


class RNNCell:
    """standard RNN"""
    def __init__(self, i, h, o):
        """
        Initializes the RNNCell.

        Parameters:
        - i: Dimensionality of the input data.
        - h: Dimensionality of the hidden state.
        - o: Dimensionality of the output.
        """
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
        - h_prev: Previous hidden state, numpy.ndarray of shape (m, h).
        - x_t: Data input for the cell, numpy.ndarray of shape (m, i).

        Returns:
        - h_next: Next hidden state, numpy.ndarray of shape (m, h).
        - y: Output of the cell, numpy.ndarray of shape (m, o).
        """
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(concat_input, self.Wh) + self.bh)

        y = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = y / np.sum(y, axis=1, keepdims=True)  # Apply softmax

        return h_next, y


def rnn(rnn_cell, X, h_0):
    """
    prforms several forward propagations
    """
    H = np.array([h_0])
    Y = None
    h_n = h_0
    for x_t in X:
        h, y = rnn_cell.forward(h_n, x_t)
        H = np.append(H, h[None, :, :], axis=0)
        if not isinstance(Y, np.ndarray):
            Y = np.array([y])
        else:
            Y = np.append(Y, y[None, :, :], axis=0)
        h_n = h

    return H, Y
