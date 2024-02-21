#!/usr/bin/env python3
"""contains tenserflow stuff"""

import pandas as pd
import numpy as np

# Read the csv file
def read_csv():
    # Read the data loaded
    data_loaded = pd.read_csv('data_loaded.csv')
    data_loaded = data_loaded.to_numpy()[:]

    #split date
    discard_split_min_sec = np.array([item[1].split(':') for item in data_loaded])
    split_hour = np.array([item[0].split(' ') for item in discard_split_min_sec])
    split_date = np.array([item[0].split('-') for item in split_hour])
    split_date = split_date.astype(int)
    split_hour = split_hour[:, np.r_[1]].astype(int)
    data_loaded = np.hstack((data_loaded, split_date))
    data_loaded = np.hstack((data_loaded, split_hour))
    data_loaded = np.delete(data_loaded, 1, axis=1)

    #select columns
    data_loaded = data_loaded[:, [np.r_[2, 5, 6, 11]]]
    data_loaded = data_loaded[:, 0, :]

    #read the data_api
    APIdata = pd.read_csv('data_api.csv')
    APIdata = APIdata.to_numpy()[:]

    #split date
    discard_split_min_sec = np.array([item[1].split(':') for item in APIdata], dtype=object)
    split_hour = np.array([item[0].split(' ') for item in discard_split_min_sec])
    split_date = np.array([item[0].split('-') for item in split_hour])
    split_date = split_date.astype(int)
    split_hour = split_hour[:, np.r_[1]].astype(int)
    APIdata = np.hstack((APIdata, split_date))
    APIdata = np.hstack((APIdata, split_hour))
    APIdata = np.delete(APIdata, 1, axis=1)

    #select columns
    APIdata = APIdata[:, [np.r_[1, 2, 5, 10]]]
    APIdata = APIdata[:, 0, :]


    # fuse both
    data = np.concatenate((APIdata, data_loaded), axis = 0)
    #take percentage increase betwen open and close
    data[:, 1] = ((data[:, 1].astype(float) - data[:, 0].astype(float)) /
                  data[:, 0].astype(float)) + 1
    data = data[:,1:]
    data[1:, 1] = ((data[1:, 1].astype(float) - data[:-1, 1].astype(float)) /
                   (data[:-1, 1].astype(float) + data[1:, 1].astype(float) + 1)) + 1

    return(data)

def read_csv_api():

    #read the data_api
    APIdata = pd.read_csv('data_api.csv')
    APIdata = APIdata.to_numpy()[:]

    #split date
    discard_split_min_sec = np.array([item[1].split(':') for item in APIdata], dtype=object)
    split_hour = np.array([item[0].split(' ') for item in discard_split_min_sec])
    split_date = np.array([item[0].split('-') for item in split_hour])
    split_date = split_date.astype(int)
    split_hour = split_hour[:, np.r_[1]].astype(int)
    APIdata = np.hstack((APIdata, split_date))
    APIdata = np.hstack((APIdata, split_hour))
    APIdata = np.delete(APIdata, 1, axis=1)

    #select columns
    APIdata = APIdata[:, [np.r_[1, 2, 5, 10]]]
    APIdata = APIdata[:, 0, :]


    # fuse both
    data = APIdata
    #take percentage increase betwen open and close
    data[:, 1] = ((data[:, 1].astype(float) - data[:, 0].astype(float)) /
                  data[:, 0].astype(float)) + 1
    data = data[:,1:]
    data[1:, 1] = ((data[1:, 1].astype(float) - data[:-1, 1].astype(float)) /
                   (data[:-1, 1].astype(float) + data[1:, 1].astype(float) + 1)) + 1

    return(data)
data = read_csv()
df = pd.DataFrame(data)
df.to_csv('data_look.csv', index=False)
# print(data)
# print(np.max(data.T[0]), np.min(data.T[0]))
# print(np.max(data.T[1]), np.min(data.T[1]))
