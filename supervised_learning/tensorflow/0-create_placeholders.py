#!/usr/bin/env python3
import tensorflow as tf

tf
def create_placeholders(nx, classes):
    """ returns two placeholders, x and y, for the neural network"""
    x = tf.compat.v1.placeholder(tf.int, shape=(1, nx))
    y = tf.compat.v1.placeholder(tf.int, shape=(classes, nx))
    return x, y
