#!/usr/bin/env python3
"""contains tenserflow stuff"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Arguments:
    y -- placeholder for the labels of the input data
    y_pred -- tensor containing the network's predictions

    Returns:
    accuracy -- tensor containing the decimal accuracy of the prediction
    """

    # Calculate the number of correct predictions
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    # Calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
