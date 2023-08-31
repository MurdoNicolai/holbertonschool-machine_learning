#!/usr/bin/env python3
"""contains tenserflow stuff"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.

    Arguments:
    loss -- the loss of the network's prediction
    alpha -- the learning rate

    Returns:
    train_op -- an operation that trains the network using gradient descent
    """

    # Create the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    # Create the training operation
    train_op = optimizer.minimize(loss)

    return train_op
