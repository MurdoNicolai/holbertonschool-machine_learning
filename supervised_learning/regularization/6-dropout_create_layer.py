#!/usr/bin/env python3
"""containst reg_cost functino"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Create a TensorFlow layer with L2 regularization.

    Returns:
    The output of the new layer.
    """
    dense_layer = tf.layers.dense(prev, units=n, activation=activation,
                                  kernel_initializer=tf.
                                  keras.initializers.
                                  VarianceScaling(scale=2.0, mode=("fan_avg")))
    return tf.compat.v1.layers.Dropout(1 - keep_prob)(dense_layer)
