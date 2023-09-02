#!/usr/bin/env python3
""" moving average"""
import numpy as np
import tensorflow as tf

def create_momentum_op(loss, alpha, beta1):
    """
    Create the momentum optimization operation for a neural network.

    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    momentum_op = optimizer.minimize(loss)

    return momentum_op
