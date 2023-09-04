#!/usr/bin/env python3
"""contains create_confusion_matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix"""
    CCM = np.zeros((len(labels[0]), len(logits[0])))
    for nb in range(len(labels)):
        CCM += np.matmul(labels[nb][np.newaxis].T, logits[nb][np.newaxis])
    return CCM
