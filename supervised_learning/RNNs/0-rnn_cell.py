#!/usr/bin/env python3
"""contains classes for rnn"""
import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        """
        Initializes the RNNCell.

        Parameters:
        - i: Dimensionality of the input data.
        - h: Dimensionality of the hidden state.
        - o: Dimensionality of the output.
        """
        # Initialize weights with random normal distribution
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

        # Initialize biases as zeros
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
        # Concatenate previous hidden state and input data
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Compute hidden state using tanh activation function
        h_next = np.tanh(np.dot(concat_input, self.Wh) + self.bh)

        # Compute output using softmax activation function
        y = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = y / np.sum(y, axis=1, keepdims=True)  # Apply softmax

        return h_next, y
