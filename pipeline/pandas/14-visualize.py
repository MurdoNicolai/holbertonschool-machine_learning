#!/usr/bin/env python3

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the data
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the column 'Weighted_Price'
df = df.drop(columns=['Weighted_Price'])

# Rename the column 'Timestamp' to 'Date'
df = df.rename(columns={'Timestamp': 'Date'})

# Convert the timestamp values to datetime values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the data frame on 'Date'
df = df.set_index('Date')

# Fill missing values as specified
df['Close'].fillna(method='ffill', inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# Filter data from 2017 and beyond
df = df[df.index >= '2017-01-01']

# Resample the data at daily intervals and aggregate
daily_df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(daily_df.index, daily_df['High'], label='High')
plt.plot(daily_df.index, daily_df['Low'], label='Low')
plt.plot(daily_df.index, daily_df['Open'], label='Open')
plt.plot(daily_df.index, daily_df['Close'], label='Close')
plt.legend()
plt.title('Daily Price Data')
plt.xlabel('Date')
plt.ylabel('Price')

plt.subplot(3, 1, 2)
plt.plot(daily_df.index, daily_df['Volume_(BTC)'], label='Volume (BTC)', color='orange')
plt.legend()
plt.title('Daily Volume (BTC)')
plt.xlabel('Date')
plt.ylabel('Volume (BTC)')

plt.subplot(3, 1, 3)
plt.plot(daily_df.index, daily_df['Volume_(Currency)'], label='Volume (Currency)', color='green')
plt.legend()
plt.title('Daily Volume (Currency)')
plt.xlabel('Date')
plt.ylabel('Volume (Currency)')

plt.tight_layout()
plt.show()

