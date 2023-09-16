#!/usr/bin/env python3
""" has the build model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""

    model = K.Sequential()
    first_layer = True
    for layer, activation in zip(layers, activations):
        if first_layer:
            model.add(K.layers.Dense(layer, activation=activation,
                                     input_shape=(nx,),
                                     kernel_regularizer=K.regularizers.
                                     l2(lambtha)))
            first_layer = False
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layer, activation=activation,
                                     input_shape=(nx,),
                                     kernel_regularizer=K.regularizers.
                                     l2(lambtha)))
    return model
