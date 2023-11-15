#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means:
        X - the dataset
        k - number of clusters
    """
    return np.random.uniform(np.min(X, axis = 0), np.max(X, axis = 0), (k, X.shape[1]))
