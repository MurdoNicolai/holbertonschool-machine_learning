#!/usr/bin/env python3
""" contains normalization"""
import numpy as np


def normalization_constants(X):
    """"Returns the normalization constant"""
    return(np.average(X, axis=0), np.std(X, axis=0))
