#!/usr/bin/env python3
import numpy as np

# Create a 4x3x4x2 matrix (assuming 'matrix' is your matrix)
matrix = np.zeros((4, 3, 4, 2))  # Example random matrix

# Create a 4x1x2 value (assuming 'value' is your value)
value = np.array([[1,2],[3,4],[5,6],[7,8]])  # Example random value

# Assign the value to the specified positions (allx2x2xall)
matrix[:, 2, 2, :] = value

print(matrix[0][0])
print(matrix[0][2])
print(matrix[1][0])
print(matrix[1][2])
