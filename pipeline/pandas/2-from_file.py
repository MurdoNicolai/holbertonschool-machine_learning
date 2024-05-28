#!/usr/bin/env python3
import pandas as pd

def from_file(filename, delimiter):
    """
    Load data from a file as a pd.DataFrame.

    Args:
    filename: The file to load from.
    delimiter: The column separator.

    Returns:
    df: The loaded pd.DataFrame.
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df

# Example usage:
# Assuming you have a file 'data.csv' with comma-separated values
# df = from_file('data.csv', ',')
# print(df)
