#!/usr/bin/env python3
"""contains tenserflow stuff"""

import pandas as pd
import numpy as np

# Read the csv file
def read_csv():
    data = pd.read_csv('data.csv')
    data = data.to_numpy()[1:]

    #split date
    discard_split_min_sec = np.array([item[1].split(':') for item in data])
    split_hour = np.array([item[0].split(' ') for item in discard_split_min_sec])
    split_date = np.array([item[0].split('-') for item in split_hour])
    split_date = split_date.astype(int)
    split_hour = split_hour[:, np.r_[1]].astype(int)
    data = np.hstack((data, split_date))
    data = np.hstack((data, split_hour))
    data = np.delete(data, 1, axis=1)

    #select columns
    data = data[:, [np.r_[2, 5, 6, 11]]]
    data = data[:, 0, :]

    #take percentage increase betwen open and close
    print(data)
    data[:, 1] = ((data[:, 1].astype(float) - data[:, 0].astype(float)) /
                  data[:, 0].astype(float)) + 1
    data = data[:,1:]
    data[1:, 1] = ((data[1:, 1].astype(float) - data[:-1, 1].astype(float)) /
                   (data[:-1, 1].astype(float) + data[1:, 1].astype(float) + 1)) + 1
    data = data[1:]
    return(data)
data = read_csv()
print(data)
print(np.max(data.T[0]), np.min(data.T[0]))
print(np.max(data.T[1]), np.min(data.T[1]))
