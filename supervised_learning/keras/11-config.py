#!/usr/bin/env python3
""" has optimize_model"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration to a file.
    """
    with open(filename, 'w') as config_file:
        config_file.write(network.to_yaml())


def load_config(filename):
    """
    Loads a model with a specific configuration from a file.
    """
    with open(filename, 'r') as config_file:
        loaded_model = K.models.model_from_yaml(config_file.read())
    return loaded_model
