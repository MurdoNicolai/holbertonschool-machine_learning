#!/usr/bin/env python3

import numpy as np
kmeans = __import__('1-kmeans').kmeans

if __name__ == "__main__":
    print(kmeans('hello', 5))
    print(kmeans(np.array([1, 2, 3, 4, 5]), 5))
    print(kmeans(np.array([[[1, 2, 3, 4, 5]]]), 5))
