#!/usr/bin/env python3
"""contains specificity"""
import numpy as np


def specificity(confusion):
    """retrun specificity of confusion matrix"""
    result = np.zeros(len(confusion))
    all_accurate = 0
    all = np.sum(confusion)
    for nb in range(len(confusion)):
        all_accurate += confusion[nb][nb]
    for nb in range(len(confusion)):
        true_negatives = (all + confusion[nb][nb]
                          - np.sum(confusion[nb]) - np.sum(confusion.T[nb]))
        true_positivs = confusion[nb][nb]
        false_postives = np.sum(confusion.T[nb]) - true_positivs
        result[nb] = true_negatives/(false_postives + true_negatives)
    return result
