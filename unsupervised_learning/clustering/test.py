#!/usr/bin/env python3

import numpy as np
pdf = __import__('5-pdf').pdf

if __name__ == "__main__":
    m = np.random.randn(6)
    S = np.random.randn(6, 6)
    print(pdf('hello', m, S))
    print(pdf(np.array([1, 2, 3, 4, 5]), m, S))
    print(pdf(np.array([[[1, 2, 3, 4, 5]]]), m, S))
