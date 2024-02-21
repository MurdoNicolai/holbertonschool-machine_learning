#!/usr/bin/env python3
"""contains tenserflow stuff"""

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.initializers import GlorotUniform
from preprocess_data import read_csv, read_csv_api
from tensorflow.keras import backend as K



data = read_csv_api()
np.random.seed(42)
datadays = -(-len(data)//24) + 1
newdata = np.random.choice([0.999999, 1.000001], size=(datadays, 2, 24))
newrow = 0
for row in data:
    newdata[newrow, :, row[2]]=row[:2]
    if row[2] == 23:
        newrow += 1
data = newdata
training = data[:(len(data) - 1)]

answers = data[1:(len(data))]
firstQ = np.average(np.quantile(answers, 0.25, axis = 0)[0])

lastQ = np.average(np.quantile(answers, 0.75, axis = 0)[0])

#answers each hour has 2 numbers telling if they are in first/last q or not thus 48 not 24 (Example: for 8 am. value 7 indicts if it corresponds to a big decreas frome 7am, and value 31 (7+24) indicats if it's a big increase)
new_answers = np.zeros((answers.shape[0], 48))
for row in range(len(answers)):
    for h in range(24):
        if answers[row][0][h] < firstQ:
            new_answers[row][h] = 1
        elif answers[row][0][h] > lastQ:
            new_answers[row][h + 24] = 1


answers = new_answers
#create the model

# Split the data into training and validation sets
validation_split = 0.1
test_split = 0.1
validation_samples = int(training.shape[0] * validation_split)
test_samples = int(training.shape[0] * test_split)


X_train = training[:-(validation_samples + test_samples)]
y_train = answers[:-(validation_samples + test_samples)]
X_val = training[-(validation_samples + test_samples):-test_samples]
y_val = answers[-(validation_samples + test_samples):-test_samples]
X_test = training[-test_samples:]
y_test = answers[-test_samples:]

## Test the model
model_path = './my_model1.keras'

# Load the model
model = tf.keras.models.load_model(model_path)
predictions = model.predict(X_test)
answers = y_test

# reorganise data by hour ignoring days

predictions2 = np.zeros((predictions.shape[0]*24, 2))
new_i = 0
for row in predictions:
    for h in range(24):
        predictions2[new_i][0] = row[h]
        predictions2[new_i][1] = row[h+ 24]
        new_i += 1
predictions = predictions2

answers2 = np.ones((answers.shape[0]*24))

new_i = 0
for row in answers:
    for h in range(24):
        answers2[new_i] += row[h+ 24]
        answers2[new_i] -= row[h]
        new_i += 1
answers = answers2

input = predictions
answers = answers
max = np.max(input) * 2
min = np.min(input) * 2
best = 0
x = -0.3
orrigin = (max)*(-x) + 0.5
new_predictions = np.ones((input.shape[0]))

for row in range(len(input)):
    per_comp = orrigin + (input[row][0] + input[row][1]) * x
    if (input[row][0]/(input[row][0] + input[row][1])) >= per_comp:
        new_predictions[row] = 0
    elif (input[row][1]/(input[row][0] + input[row][1])) >= per_comp:
        new_predictions[row] = 2
correct = [0, 0, 0]
total = [0, 0, 0]
rong = [0, 0, 0, 0]

for row in range(len(new_predictions)):
    if new_predictions[row] == answers[row]:
        correct[int(answers[row])] += 1
    elif answers[row] == 1:
        rong[1] += 1
    elif new_predictions[row] == 0 and answers[2]:
        rong[0] += 1
    elif new_predictions[row] == 2 and answers[0]:
        rong[2] += 1
    else:
        rong[3] += 1
    total[int(answers[row])] += 1

div = correct[0] - rong[0] + correct[2] - rong[2]
result = 100*(1 + div / (correct[0] + correct[2]))
print(correct, rong)
print(result)
print(div)
