#!/usr/bin/env python3
"""contains tenserflow stuff"""
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for a neural network.

    Arguments:
    x -- placeholder for input data
    layer_sizes -- list of layer sizes (number of nodes in each layer)
    activations -- list of activation functions for each layer

    Returns:
    prediction -- tensor representing the prediction of the network
    """

    # Import create_layer function
    create_layer = __import__('1-create_layer').create_layer

    prev_layer = x  # Initialize the previous layer with input data

    # Iterate over layer sizes and activations to create layers
    for i in range(len(layer_sizes)):
        size = layer_sizes[i]
        activation = activations[i]
        prev_layer = create_layer(prev_layer, size, activation)

    prediction = prev_layer  # The final layer's output is the prediction

    return prediction
