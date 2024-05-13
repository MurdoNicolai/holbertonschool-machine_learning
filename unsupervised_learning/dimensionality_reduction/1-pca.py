#!/usr/bin/env python3
"""contains PCA functions"""
import numpy as np


def pca(X, ndim):
    '''Performs PCA on a dataset'''
    # Calculate the mean of each feature
    mean = np.mean(X, axis=0)

    # Center the data
    centered_data = X - mean

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'ndim' eigenvectors
    top_eigenvectors = -sorted_eigenvectors[:, :ndim]

    # Project the data onto the new subspace
    transformed_data = np.dot(centered_data, top_eigenvectors)

    return transformed_data
