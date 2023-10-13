#!/usr/bin/env python3
"""identity_block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Implement the identity block as defined in Deep Residual
    Learning for Image Recognition (2015).

    Args:
    A_prev (tf.Tensor): Output from the previous layer.
    filters (tuple or list): A tuple containing F11, F3, and F12, respectively:
        F11: Number of filters in the first 1x1 convolution.
        F3: Number of filters in the 3x3 convolution.
        F12: Number of filters in the second 1x1 convolution.

    Returns:
    tf.Tensor: The activated output of the identity block.
    """
    F11, F3, F12 = filters
    x = A_prev
    # First 1x1 convolution layer
    x = K.layers.Conv2D(F11, (1, 1), strides=(1, 1), padding='valid',
                        kernel_initializer=K.initializers.he_normal(seed=0))(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    # 3x3 convolution layer
    x = K.layers.Conv2D(F3, (3, 3), strides=(1, 1), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0))(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    # Second 1x1 convolution layer
    x = K.layers.Conv2D(F12, (1, 1), strides=(1, 1), padding='valid',
                        kernel_initializer=K.initializers.he_normal(seed=0))(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    # Add the shortcut (input) to the output
    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)

    return x
