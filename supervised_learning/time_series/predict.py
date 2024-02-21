#!/usr/bin/env python3
"""contains tenserflow stuff"""

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.initializers import GlorotUniform
from preprocess_data import read_csv, read_csv_api
from tensorflow.keras import backend as K

@keras.saving.register_keras_serializable()
def loss_with_entropy_regularization(y_true, y_pred):
        # Compute the binary crossentropy loss
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Add the entropy regularization penalty
        entropy = loss - tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10))
        penalty = EntropyRegularizer(rate=0.01)(y_true, y_pred)

        return loss + penalty

def predict():
    """ returns the prediction of big bitcoin price increases/decreses in the floowing 24 h and the latest hour"""
    data = read_csv_api()
    last_hour = data[0][2]
    np.random.seed(42)
    datadays = -(-len(data)//24) + 1
    newdata = np.random.choice([0.999999, 1.000001], size=(datadays, 2, 24))
    newrow = 0
    for row in data:
        newdata[newrow, :, row[2]]=row[:2]
        if row[2] == 23:
            newrow += 1
    data = newdata

    ## Test the model
    model_path = './my_model_api_rand3.keras'

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(data)[-1]

    hours = np.zeros((24,2))
    for h in range(24):
        hours[h][0] = predictions[h]
        hours[h][1] = predictions[h+ 24]


    input = hours
    max = np.max(input) * 2
    min = np.min(input) * 2

    x = -0.3
    orrigin = (max)*(-x) + 0.5

    predictions = np.ones((input.shape[0]))
    for row in range(len(input)):
        per_comp = orrigin + (input[row][0] + input[row][1]) * x
        if (input[row][0]/(input[row][0] + input[row][1])) >= per_comp:
            predictions[row] = 0
        elif (input[row][1]/(input[row][0] + input[row][1])) >= per_comp:
            predictions[row] = 2
    return(predictions, last_hour)
print(predict())
