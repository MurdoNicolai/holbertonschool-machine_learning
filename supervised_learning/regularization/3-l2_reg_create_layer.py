#!/usr/bin/env python3
"""containst reg_cost functino"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Create a TensorFlow layer with L2 regularization.

    Returns:
    The output of the new layer.
    """
    layer_name = "layer_with_l2_reg"

    # Define the weight matrix with L2 regularization
    with tf.variable_scope(layer_name):
        initializer = tf.random_normal_initializer(stddev=0.1)
        weights = tf.get_variable("weights", shape=[int(prev.get_shape()[1]), n],
                                  initializer=initializer,
                                  regularizer=tf.keras.regularizers.L2(lambtha))

        # Define the bias vector
        bias = tf.get_variable("bias", shape=[n], initializer=tf.zeros_initializer())

    # Linear combination of inputs and weights
    z = tf.matmul(prev, weights) + bias

    # Apply activation function
    if activation is not None:
        dense_layer = activation(z)
    else:
        dense_layer = z

    return dense_layer
