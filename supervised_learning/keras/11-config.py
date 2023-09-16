#!/usr/bin/env python3
""" has optimize_model"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration to a file.
    """
    network.save(filename)


def load_config(filename):
    """
    Loads a model with a specific configuration from a file.
    """
    with open(filename, 'r') as config_file:
        loaded_model = K.models.load_model(config_file.read())
    return loaded_model
