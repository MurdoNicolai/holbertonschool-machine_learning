#!/usr/bin/env python3
""" contains modules for attention algorythms"""
import numpy as np

def positional_encoding(max_seq_len, dm):
    encoding = np.zeros((max_seq_len, dm))

    for pos in range(max_seq_len):
        for i in range(dm):
            angle = pos / np.power(10000, 2 * i / dm)
            encoding[pos, i] = np.sin(angle) if i % 2 == 0 else np.cos(angle)

    return encoding
