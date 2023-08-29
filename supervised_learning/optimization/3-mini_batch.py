#!/usr/bin/env python3
""" contains normalization"""
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """"
    trains a loaded neural network model using mini-batch gradient descent
    """
    return(0)
