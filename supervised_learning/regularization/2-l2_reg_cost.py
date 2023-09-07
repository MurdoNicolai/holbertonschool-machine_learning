#!/usr/bin/env python3
"""containst reg_cost functino"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """ calculates the cost of a neural network with L2 regularization"""
    reg_losses = tf.compat.v1.losses.get_regularization_losses()

    # Sum up the regularization losses
    total_reg_loss = tf.add_n(reg_losses)

    # Add regularization to the original loss
    total_loss = cost + total_reg_loss

    return total_loss
