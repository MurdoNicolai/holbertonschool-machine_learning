#!/usr/bin/env python3
"""to make"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Implement the projection block as defined in Deep Residual Learning for Image Recognition (2015).

    Args:
    A_prev (K.layers.Layer): Output from the previous layer.
    filters (tuple or list): A tuple containing F11, F3, and F12, respectively:
        F11: Number of filters in the first 1x1 convolution.
        F3: Number of filters in the 3x3 convolution.
        F12: Number of filters in the second 1x1 convolution, also in the shortcut connection.
    s (int): Stride for the first convolution in both the main path and the shortcut connection.

    Returns:
    K.layers.Layer: The activated output of the projection block.
    """
    F11, F3, F12 = filters

    # Shortcut path (main path will be built separately)
    shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s), padding='valid',
                               kernel_initializer='he_normal')(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Main path
    x = K.layers.Conv2D(F11, (1, 1), strides=(s, s), padding='valid',
                        kernel_initializer='he_normal')(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F3, (3, 3), strides=(1, 1), padding='same',
                        kernel_initializer='he_normal')(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    x = K.layers.Conv2D(F12, (1, 1), strides=(1, 1), padding='valid',
                        kernel_initializer='he_normal')(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    # Add the shortcut to the main path
    x = K.layers.Add()([x, shortcut])
    x = K.layers.Activation('relu')(x)

    return x

