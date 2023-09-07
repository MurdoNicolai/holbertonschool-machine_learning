#!/usr/bin/env python3
"""containst reg_cost functino"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """ calculates the cost of a neural network with L2 regularization"""
    trainable_vars = tf.trainable_variables()
    print(trainable_vars, cost)
    print([tf.nn.l2_loss(var) for var in trainable_vars])
    l2_reg_term = tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])

    cost_with_l2_reg = cost + (1 / 2.0) * l2_reg_term

    return cost_with_l2_reg

