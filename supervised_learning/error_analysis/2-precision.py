#!/usr/bin/env python3
"""contains precision"""
import numpy as np


def precision(confusion):
    """retrun precision of confusion matrix"""
    result = np.zeros(len(confusion))
    for nb in range(len(confusion)):
        positivs = confusion[nb][nb]
        supposed_positivs = np.sum(confusion.T[nb])
        result[nb] = positivs/supposed_positivs
    return result
