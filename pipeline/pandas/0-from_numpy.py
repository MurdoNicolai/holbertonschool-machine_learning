#!/usr/bin/env python3
import numpy as np
import pandas as pd

def from_numpy(array):
    """
    Create a pd.DataFrame from a np.ndarray.

    Args:
    array: The np.ndarray from which to create the pd.DataFrame.

    Returns:
    df: The newly created pd.DataFrame with columns labeled in alphabetical order and capitalized.
    """
    num_columns = array.shape[1]
    if num_columns > 26:
        raise ValueError("Array has more than 26 columns, which exceeds the limit of alphabetical labels.")

    # Generate column labels from 'A' to 'Z'
    column_labels = [chr(i) for i in range(65, 65 + num_columns)]

    # Create the DataFrame
    df = pd.DataFrame(array, columns=column_labels)

    return df

# Example usage:
# array = np.random.rand(10, 5)  # example array with 10 rows and 5 columns
# df = from_numpy(array)
# print(df)
