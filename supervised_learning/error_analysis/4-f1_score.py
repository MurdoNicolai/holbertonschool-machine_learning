#!/usr/bin/env python3
"""contains f1_score"""
import numpy as np


def f1_score(confusion):
    """retrun f1_score of confusion matrix"""
    result = np.zeros(len(confusion))
    for nb in range(len(confusion)):
        positivs = confusion[nb][nb]
        supposed_positivs = np.sum(confusion.T[nb])
        actual_positivs = np.sum(confusion[nb])
        result[nb] = 2 * positivs/(supposed_positivs + actual_positivs)
    return result
