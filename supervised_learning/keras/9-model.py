#!/usr/bin/env python3
""" has optimize_model"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a file.

    Args:
        network (keras.Model): The model to save.
        filename (str): The path of the file to save the model to.

    Returns:
        None
    """
    network.save(filename)

def load_model(filename):
    """
    Loads an entire Keras model from a file.

    Args:
        filename (str): The path of the file to load the model from.

    Returns:
        keras.Model: The loaded Keras model.
    """
    loaded_model = K.models.load_model(filename)
    return loaded_model
