#!/usr/bin/env python3
import numpy as np

# Assuming newdata is a NumPy array
newdata = np.array([['1706716800', '40229.19', '40229.19', '40229.19', '40229.19', '0.0005', '20.114595']])

# Concatenate strings "a", "b", "c", and "d"
concatenated_str = "a" + "b" + "c" + "d"

# Insert the concatenated string into the array
newdata = np.insert(newdata, 1, "a" + "b" + "c" + "d", axis=1)

# Print the newdata array
print(newdata)
