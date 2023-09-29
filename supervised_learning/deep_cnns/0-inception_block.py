#!/usr/bin/env python3
""" contains deep convolution neural network function"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ builds an inception block """
    x = A_prev
    # 1x1 Convolution
    conv1x1 = K.layers.Conv2D(filters[0], (1, 1), padding='same',
                              activation='relu')(x)

    # 3x3 Convolution
    conv3x3 = K.layers.Conv2D(filters[1], (3, 3), padding='same',
                              activation='relu')(x)

    # 5x5 Convolution
    conv5x5 = K.layers.Conv2D(filters[2], (5, 5), padding='same',
                              activation='relu')(x)

    # Max Pooling
    max_pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    max_pool_conv = K.layers.Conv2D(filters[3], (1, 1), padding='same',
                                    activation='relu')(max_pool)

    # Concatenate all the paths
    inception_output = K.layers.Concatenate(axis=-1)([conv1x1, conv3x3,
                                                      conv5x5, max_pool_conv])

    return inception_output


