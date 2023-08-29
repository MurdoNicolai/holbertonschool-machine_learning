#!/usr/bin/env python3
""" contains normalization"""
import numpy as np


def normalize(X, m, s):
    """"Returns the normalization constant"""
    return((X - m)/s)
