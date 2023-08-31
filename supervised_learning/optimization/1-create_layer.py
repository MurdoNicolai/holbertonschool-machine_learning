#!/usr/bin/env python3
"""contains tenserflow stuff"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Arguments:
    prev -- tensor output of the previous layer
    n -- number of nodes in the layer
    activation -- activation function for the layer

    Returns:
    tensor output of the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')
    return layer(prev)
