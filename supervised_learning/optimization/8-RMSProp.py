#!/usr/bin/env python3
""" moving average"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Create the RMSProp optimization operation for a neural network in TensorFlow.
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)

    return train_op
