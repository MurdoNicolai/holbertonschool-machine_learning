#!/usr/bin/env python3
"""contains PCA functions"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset"""

    cov_matrix = X.T @ X / X.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    total_variance = np.sum(eigenvalues)

    target_variance = np.sqrt(var) * total_variance

    cumulative_variance = np.cumsum(eigenvalues)
    num_components = np.argmax(cumulative_variance >= target_variance) + 1

    W = eigenvectors[:, :num_components]

    return np.real(W)

