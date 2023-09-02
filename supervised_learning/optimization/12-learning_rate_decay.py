#!/usr/bin/env python3
""" moving average"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Create a learning rate decay operation in TensorFlow using inverse time decay.
    """
    learning_rate = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate
    )
    return learning_rate
