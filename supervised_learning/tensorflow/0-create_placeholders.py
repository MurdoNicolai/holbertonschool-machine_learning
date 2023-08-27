#!/usr/bin/env python3
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ returns two placeholders, x and y, for the neural network"""
    x = tf.placeholder("int", shape=(1, nx))
    y = tf.v1.placeholder("int", shape=(classes, nx))
    return x, y
