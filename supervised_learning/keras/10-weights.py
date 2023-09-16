#!/usr/bin/env python3
""" has optimize_model"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a model's weights to a file.

    Args:
        network (keras.Model): The model whose weights should be saved.
        filename (str): The path of the file to save the weights to.
        save_format (str): The format in which the weights should be saved (default is 'h5').

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)

def load_weights(network, filename):
    """
    Loads a model's weights from a file.

    Args:
        network (keras.Model): The model to which the weights should be loaded.
        filename (str): The path of the file to load the weights from.

    Returns:
        None
    """
    network.load_weights(filename)
