#!/usr/bin/env python3
""" has the build model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""

    input_layer = K.layers.Input(shape=(nx,))

    x = input_layer

    for layer, activation in zip(layers, activations):
        x = K.layers.Dense(
            layer, activation=activation,
            kernel_regularizer=K.regularizers.l2(lambtha))(x)

        x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.models.Model(inputs=input_layer, outputs=x)

    return model
