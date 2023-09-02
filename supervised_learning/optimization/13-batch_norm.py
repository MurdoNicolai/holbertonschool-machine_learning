#!/usr/bin/env python3
""" moving average"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalize an unactivated output of a neural network using batch normalization.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_normalized = gamma * Z_norm + beta
    return Z_normalized
