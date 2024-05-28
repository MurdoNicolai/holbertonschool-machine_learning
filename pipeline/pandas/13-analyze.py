#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Drop the 'Timestamp' column
df = df.drop(columns=['Timestamp'])

# Calculate descriptive statistics for all columns
stats = df.describe()

print(stats)

