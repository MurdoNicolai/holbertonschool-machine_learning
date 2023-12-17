#!/usr/bin/env python3
"""contains classes for rnn"""
import numpy as np


class BidirectionalCell:
    """Bidirectional RNN Cell"""
    def __init__(self, i, h, o):
        """
        Initializes the BidirectionalCell.

        Parameters:
        - i: Dimensionality of the input data.
        - h: Dimensionality of the hidden states.
        - o: Dimensionality of the outputs.
        """
        # Forward direction parameters
        self.Whf = np.random.normal(size=(h + i, h))
        self.bhf = np.zeros(shape=(1, h))

        # Backward direction parameters
        self.Whb = np.random.normal(size=(h + i, h))
        self.bhb = np.zeros(shape=(1, h))

        # Output parameters
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.

        Parameters:
        - h_prev: Previous hidden state, numpy.ndarray of shape (m, h).
        - x_t: Data input for the cell, numpy.ndarray of shape (m, i).

        Returns:
        - h_next: Next hidden state, numpy.ndarray of shape (m, h).
        """
        # Forward direction
        concat_input_forward = np.concatenate((h_prev, x_t), axis=1)
        h_next_forward = np.tanh(np.dot(concat_input_forward, self.Whf) +
                                 self.bhf)

        return h_next_forward

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step

        Parameters:
        - h_next: Next hidden state, numpy.ndarray of shape (m, h).
        - x_t: Data input for the cell, numpy.ndarray of shape (m, i).

        Returns:
        - h_prev: Previous hidden state, numpy.ndarray of shape (m, h).
        """
        # Backward direction
        concat_input_backward = np.concatenate((h_next, x_t), axis=1)
        h_prev_backward = np.tanh(np.dot(concat_input_backward, self.Whb) +
                                  self.bhb)

        return h_prev_backward

    def output(self, H):
        """
        Calculates all outputs for the RNN.

        Parameters:
        - H: Concatenated hidden states from both directions
            Excludes their initialized states.
            t is the number of time steps.
            m is the batch size for the data.
            h is the dimensionality of the hidden states.

        Returns:
        - Y: Outputs, numpy.ndarray of shape (t, m, o).
        """
        t, m, _ = H.shape
        Y = np.zeros((t, m, self.Wy.shape[1]))

        for t_step in range(t):
            h_concat = H[t_step]
            y = np.dot(h_concat, self.Wy) + self.by
            Y[t_step] = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return Y
