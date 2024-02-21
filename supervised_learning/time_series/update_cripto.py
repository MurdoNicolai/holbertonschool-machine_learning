#!/usr/bin/env python3
"""connects to kucoin api"""

import os
from kucoin.client import Client
import numpy as np
import pandas as pd
from api_to_csv import update_csv

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
