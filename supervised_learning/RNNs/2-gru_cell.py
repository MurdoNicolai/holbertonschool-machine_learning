#!/usr/bin/env python3
"""contains classes for rnn"""
import numpy as np


class GRUCell:
    """Gated Recurrent Unit (GRU) Cell"""
    def __init__(self, i, h, o):
        """
        Initializes the GRUCell.

        Parameters:
        - i: Dimensionality of the input data.
        - h: Dimensionality of the hidden state.
        - o: Dimensionality of the output.
        """
        # Update gate parameters
        self.Wz = np.random.normal(size=(h + i, h))
        self.bz = np.zeros(shape=(1, h))

        # Reset gate parameters
        self.Wr = np.random.normal(size=(h + i, h))
        self.br = np.zeros(shape=(1, h))

        # Intermediate hidden state parameters
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros(shape=(1, h))

        # Output parameters
        self.Wy = np.random.normal(size=(h, o))
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
        # Update gate
        z_t = sigmoid(np.dot(np.concatenate((h_prev, x_t), axis=1), self.Wz)
                      + self.bz)

        # Reset gate
        r_t = sigmoid(np.dot(np.concatenate((h_prev, x_t), axis=1), self.Wr)
                      + self.br)

        # Intermediate hidden state
        h_tilde = np.tanh(np.dot(np.concatenate((r_t * h_prev, x_t), axis=1),
                                 self.Wh) + self.bh)

        # Final hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output
        y = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = y / np.sum(y, axis=1, keepdims=True)  # Apply softmax

        return h_next, y


# Helper function for sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
