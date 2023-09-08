#!/usr/bin/env python3
"""containst reg_cost functino"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Create a TensorFlow layer with L2 regularization.

    Returns:
    The output of the new layer.
    """
    dense_layer = tf.layers.dense(units=n, activation=activation,
                                  kernel_initializer=tf.
                                  keras.initializers.
                                  VarianceScaling(scale=2.0, mode=("fan_avg")),
                                  kernel_regularizer=tf.
                                  keras.regularizers.L2(lambtha))(prev)
    return dense_layer
