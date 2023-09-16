#!/usr/bin/env python3
""" has optimize_model"""
import numpy as np


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.
    """
    if classes is None:
        classes = np.max(labels) + 1

    one_hot_matrix = np.eye(classes)[labels]

    return one_hot_matrix
