#!/usr/bin/env python3
"""containst reg_cost functino"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Create a TensorFlow layer with L2 regularization.

    Returns:
    The output of the new layer.
    """
    print(lambtha)
    dense_layer = tf.layers.dense(prev, units=n, activation=activation,
                                  kernel_regularizer=tf.keras.
                                  regularizers.L2(lambtha*2.5))
    return dense_layer
