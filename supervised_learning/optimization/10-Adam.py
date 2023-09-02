#!/usr/bin/env python3
""" moving average"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Create the training operation using the Adam optimization algorithm.
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    adam_op = optimizer.minimize(loss)

    return adam_op
