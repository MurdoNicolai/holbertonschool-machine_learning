#!/usr/bin/env python3
"""contains classes for rnn"""
import numpy as np


class LSTMCell:
    """Long Short-Term Memory (LSTM) Cell"""
    def __init__(self, i, h, o):
        """
        Initializes the LSTMCell.

        Parameters:
        - i: Dimensionality of the input data.
        - h: Dimensionality of the hidden state.
        - o: Dimensionality of the output.
        """
        # Forget gate parameters
        self.Wf = np.random.normal(size=(h + i, h))
        self.bf = np.zeros(shape=(1, h))

        # Update gate parameters
        self.Wu = np.random.normal(size=(h + i, h))
        self.bu = np.zeros(shape=(1, h))

        # Cell state parameters
        self.Wc = np.random.normal(size=(h + i, h))
        self.bc = np.zeros(shape=(1, h))

        # Output gate parameters
        self.Wo = np.random.normal(size=(h + i, h))
        self.bo = np.zeros(shape=(1, h))

        # Output parameters
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
        - h_prev: Previous hidden state, numpy.ndarray of shape (m, h).
        - c_prev: Previous cell state, numpy.ndarray of shape (m, h).
        - x_t: Data input for the cell, numpy.ndarray of shape (m, i).

        Returns:
        - h_next: Next hidden state, numpy.ndarray of shape (m, h).
        - c_next: Next cell state, numpy.ndarray of shape (m, h).
        - y: Output of the cell, numpy.ndarray of shape (m, o).
        """
        # Forget gate
        f_t = sigmoid(np.dot(np.concatenate((h_prev, x_t), axis=1),
                             self.Wf) + self.bf)

        # Update gate
        u_t = sigmoid(np.dot(np.concatenate((h_prev, x_t), axis=1),
                             self.Wu)
                      + self.bu)

        # Intermediate cell state
        c_tilde = np.tanh(np.dot(np.concatenate((h_prev, x_t), axis=1),
                                 self.Wc)
                          + self.bc)

        # New cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # Output gate
        o_t = sigmoid(np.dot(np.concatenate((h_prev, x_t), axis=1), self.Wo)
                      + self.bo)

        # New hidden state
        h_next = o_t * np.tanh(c_next)

        # Output
        y = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = y / np.sum(y, axis=1, keepdims=True)  # Apply softmax

        return h_next, c_next, y


# Helper function for sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
