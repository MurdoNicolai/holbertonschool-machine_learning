#!/usr/bin/env python3
"""contains sensitivity"""
import numpy as np


def sensitivity(confusion):
    """creates a confusion matrix"""
    result = np.zeros(len(confusion))
    for nb in range(len(confusion)):
        positivs = confusion[nb][nb]
        predicted_positivs = np.sum(confusion[nb])
        result[nb] = positivs/predicted_positivs
    return result
