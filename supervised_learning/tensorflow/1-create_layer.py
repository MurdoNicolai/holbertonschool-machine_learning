#!/usr/bin/env python3
"""contains tenserflow stuff"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a fully connected layer with specified activation.

    Arguments:
    prev -- tensor output of the previous layer
    n -- number of nodes in the layer to create
    activation -- activation function to use for the layer

    Returns:
    layer -- tensor output of the layer
    """

    # He initialization for the layer weights
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create the layer with specified number of nodes and activation
    layer = tf.layers.dense(prev, units=n, activation=activation,
                            kernel_initializer=initializer,
                            name="layer")

    return layer
