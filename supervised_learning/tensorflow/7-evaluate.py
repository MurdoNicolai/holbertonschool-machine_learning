#!/usr/bin/env python3
"""contains tenserflow stuff"""
import tensorflow.compat.v1 as tf
import numpy as np


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Arguments:
    X -- numpy.ndarray containing the input data to evaluate
    Y -- numpy.ndarray containing the one-hot labels for X
    save_path -- location to load the model from

    Returns:
    prediction -- network's prediction
    accuracy -- accuracy of the prediction
    loss -- loss of the prediction
    """

    # Import necessary functions
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy

    # Load the model
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Get tensors from the graph's collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]

        # Evaluate the model on the input data
        prediction, eval_loss, eval_accuracy = sess.run([y_pred, loss, accuracy], feed_dict={x: X, y: Y})

    return prediction, eval_accuracy, eval_loss
