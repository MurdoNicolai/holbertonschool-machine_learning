#!/usr/bin/env python3
"""contains matrix math functions"""
import numpy as np


def definiteness(matrix):
    """ calculates the definiteness of a matrix """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.shape[0] != matrix.shape[1]:
        # Not a square matrix, so it's not valid
        return None

    # Check if the matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    eigvals = np.linalg.eigvals(matrix)

    # Count positive, negative, and zero eigenvalues
    pos_count = np.sum(eigvals > 0)
    neg_count = np.sum(eigvals < 0)
    zero_count = np.sum(eigvals == 0)

    if zero_count == 0:
        if pos_count == matrix.shape[0]:
            return "Positive definite"
        if neg_count == matrix.shape[0]:
            return "Negative definite"
        if pos_count > 0 and neg_count > 0:
            return "Indefinite"
    elif zero_count > 0:
        if pos_count == matrix.shape[0] - zero_count:
            return "Positive semi-definite"
        if neg_count == matrix.shape[0] - zero_count:
            return "Negative semi-definite"

    return None
