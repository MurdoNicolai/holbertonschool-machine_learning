#!/usr/bin/env python3
import pandas as pd

# Load the data from a file as a DataFrame
def from_file(filename, delimiter):
    df = pd.read_csv(filename, delimiter=delimiter)
    return df

# Load the data
filename = 'your_data_file.csv'  # replace with your actual file name
delimiter = ','  # replace with your actual delimiter if different
df = from_file(filename, delimiter)

# Rename the column 'Timestamp' to 'Datetime'
df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

# Convert the 'Datetime' column to datetime values
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Display only the 'Datetime' and 'Close' columns
result = df[['Datetime', 'Close']]

# Print the result
print(result)
