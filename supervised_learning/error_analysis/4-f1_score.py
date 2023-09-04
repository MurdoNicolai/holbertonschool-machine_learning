#!/usr/bin/env python3
"""contains f1_score"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """retrun f1_score of confusion matrix"""
    result = 2/(1/sensitivity(confusion) + 1/precision(confusion))
    return result
