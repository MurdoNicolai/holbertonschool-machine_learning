#!/usr/bin/env python3
""" moving average"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    """
    # Create a Dense layer with n nodes
    dense_layer = tf.keras.layers.Dense(
        n, kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'))

    # Apply the Dense layer to the previous output
    x = dense_layer(prev)

    # Create a BatchNormalization layer
    batch_norm_layer = tf.keras.layers.BatchNormalization(epsilon=1e-8)

    # Apply the BatchNormalization layer to the output of the Dense layer
    x = batch_norm_layer(x)

    # Apply the activation function
    if activation is not None:
        x = activation(x)

    return x
