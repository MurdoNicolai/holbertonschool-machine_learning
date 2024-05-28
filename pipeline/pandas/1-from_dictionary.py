#!/usr/bin/env python3
import pandas as pd

# Define the data for the DataFrame
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Create the DataFrame with specified row labels
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])

# Display the DataFrame
print(df)
