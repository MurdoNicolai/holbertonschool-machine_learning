#!/usr/bin/env python3
"""contains specificity"""
import numpy as np


def specificity(confusion):
    """retrun specificity of confusion matrix"""
    result = np.zeros(len(confusion))
    all_accurate = 0
    for nb in range(len(confusion)):
        all_accurate += confusion[nb][nb]
    for nb in range(len(confusion)):
        true_negatives = all_accurate - confusion[nb][nb]
        positivs = confusion[nb][nb]
        supposed_positivs = np.sum(confusion[nb])
        all_negatives = supposed_positivs - positivs + true_negatives
        result[nb] = true_negatives/all_negatives
    return result
