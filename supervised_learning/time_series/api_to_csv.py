#!/usr/bin/env python3
"""connects to kucoin api"""

import os
from datetime import datetime
from kucoin.client import Client
import numpy as np
import pandas as pd
from tqdm import tqdm


# get currencies
# currencies = client.get_currencies()
# for currencie in currencies:
#     if currencie["name"] == "BTC":
#         print(currencie)

# get symbol
# symbols = client.get_symbols()
# for symbol in symbols:
#     if symbol["symbol"] == "BTC-EUR":
#         print(symbol)


def update(client):
    data = pd.read_csv('data_api.csv')
    data = data.to_numpy()
    utc_start = (data[4][0])#load +5 lines of data for no errors
    print(utc_start)
    newdata = client.get_kline_data('BTC-EUR', '1hour', utc_start)

    if newdata == []:
        return data

    newdata = np.array(newdata)
    timestamps = newdata[:, 0]
    print(timestamps)
    datetime_objects = np.array([datetime.utcfromtimestamp(int(ts)) for ts in timestamps])

    newdata = np.insert(newdata, 1, datetime_objects, axis=1)

    # print(newdata, datetime_objects)

    for row in range(5): #delete 5 extra lines loaded
        data = np.delete(data, 0, axis=0)

    data = np.concatenate((newdata, data), axis = 0)
    return (data)

def create_data_api(client):
    data = []
    utc_start = 1526364000
    utc_last = int(datetime.timestamp(datetime.utcnow()))
    while utc_start < utc_last:
        data = client.get_kline_data("BTC-EUR", '1hour', utc_start, utc_start + 540000) + data
        utc_start = utc_start + 540000
    data = np.array(data)
    timestamps = data[:, 0]
    datetime_objects = np.array([datetime.utcfromtimestamp(int(ts)) for ts in timestamps])
    data = np.insert(data, 1, datetime_objects, axis=1)
    return (data)

def update_csv(client):
    """Updates the data"""

    data = update(client)


    df = pd.DataFrame(data)
    df.to_csv('data_api.csv', index=False)
    return (df)



home_directory = os.path.expanduser("~")
file_path = os.path.join(home_directory, "apicredetials")
file = open(file_path, "r").read().splitlines()

########### Credentials #################
api_key = file[0]
api_secret = file[1]
passphrase = file[2]
client = Client(api_key, api_secret, passphrase)
#########################################
update_csv(client)

# data2 = pd.read_csv('data.csv')
# data2 = data2[1:]

# data2 = np.flip(data2)
# data = np.flip(data)
# i = 0
# row = data[i]
# row2 = data2[i]
# while np. array_equal(row, row2):
#     print(row)
#     i += 1
#     row = data[i]
#     row2 = data2[i]
# print("exit\n" + row)
# print(row2)
