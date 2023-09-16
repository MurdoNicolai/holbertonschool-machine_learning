#!/usr/bin/env python3
""" has optimize_model"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes predictions using a neural network model.
    """
    predictions = network.predict(data, verbose=verbose)
    return predictions
