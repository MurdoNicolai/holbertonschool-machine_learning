#!/usr/bin/env python3
"""contains everything for Clustering"""
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means:
        X - the dataset
        k - number of clusters
    """

    if not isinstance(X, np.ndarray):
        return None

    if X.ndim != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                             (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    performs K-means on a dataset:
        X - the dataset
        k - number of clusters
    """
    C = initialize(X, k)
    if C is None:
        return (None, None)

    clss = np.zeros((X.shape[0],))
    for i in range(iterations):
        new_clusters = (np.ones((C.shape[0], 1))
                        @ np.average(C, axis=0, keepdims=True) * -1)
        new_clusters_count = np.ones(C.shape[0]) * (-1)
        for n in range(X.shape[0]):
            closest_cluster = closest(C, X[n])
            clss[n] = closest_cluster
            if new_clusters_count[closest_cluster] == -1:
                new_clusters_count[closest_cluster] = 0
                new_clusters[closest_cluster] = 0
            new_clusters[closest_cluster] += X[n]
            new_clusters_count[closest_cluster] += 1
        C = new_clusters / new_clusters_count[:, None]
    return(C, clss)


def closest(centroids, x):
    """determins the closst centroid to x"""

    closest = 0
    min_distance = 9223372036854775807
    i = 0
    for centroid in centroids:
        dist = np.linalg.norm(x - centroid)
        if dist < min_distance:
            min_distance = dist
            closest = i
        i += 1
    return (closest)
