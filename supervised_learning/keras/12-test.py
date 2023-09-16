#!/usr/bin/env python3
""" has optimize_model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network model on a dataset.
    """
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return [loss, accuracy]
