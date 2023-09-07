#!/usr/bin/env python3
"""containst reg_cost functino"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Create a TensorFlow layer with L2 regularization.

    Returns:
    The output of the new layer.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    weights = tf.get_variable("weights", shape=[int(prev.get_shape()[1]), n],
                              initializer=initializer, regularizer=regularizer)

    bias = tf.get_variable("bias", shape=[n], initializer=tf.zeros_initializer())

    z = tf.matmul(prev, weights) + bias

    if activation is not None:
        return activation(z)
    else:
        return z
