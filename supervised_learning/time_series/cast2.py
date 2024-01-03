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
from preprocess_data import read_csv
from tensorflow.keras import backend as K


class BinaryEvaluationLayer():
    def __init__(self, threshold=0.5, **kwargs):
        super(BinaryEvaluationLayer, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, inputs):
        binary_predictions = tf.cast(inputs > self.threshold, dtype=tf.float32)
        return binary_predictions

class BinaryAccuracyMetric():
    def __init__(self, **kwargs):
        super(BinaryAccuracyMetric, self).__init__(**kwargs)
        self.binary_accuracy = self.add_weight(name='binary_accuracy', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct_predictions = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32))
        total_samples = tf.cast(tf.size(y_true), dtype=tf.float32)

        self.binary_accuracy.assign_add(correct_predictions)
        self.total_samples.assign_add(total_samples)

    def result(self):
        return self.binary_accuracy / self.total_samples if self.total_samples != 0 else 0.0

def model(dropout_rate):
    model = Sequential()

    # Add the first SimpleRNN layer with input shape (2, 24)
    model.add(SimpleRNN(units=128, activation='relu', input_shape=(2, 24),
                        kernel_initializer=GlorotUniform(), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    # Add the second SimpleRNN layer with the same number of units
    # and input_shape matching the output shape of the previous layer
    model.add(SimpleRNN(units=96, activation='relu',
                        kernel_initializer=GlorotUniform()))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    # Continue with the rest of the model as before
    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=192, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=192, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=48, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.01, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Custom activation function to bias away from 1
def custom_activation(x):
    return K.sigmoid(x) * 2

#create model
def model_translator():
    model = Sequential()

    model.add(Dense(units=2, input_dim=2, activation='relu'))

    model.add(Dense(units=1, activation=custom_activation))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

data = read_csv()
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

model = model(0.5)

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
# Train the model
epochs = 10
batch_size = 32
best_result = 0

for i in range (2):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    predictions = model.predict(training)

    # reorganise data by hour ignoring days
    training = np.zeros((predictions.shape[0]*24, 2))
    new_i = 0
    for row in predictions:
        for h in range(24):
            training[new_i][0] = row[h]
            training[new_i][1] = row[h+ 24]
            new_i += 1

    answers2 = np.ones((answers.shape[0]*24))
    new_i = 0
    for row in answers:
        for h in range(24):
            answers2[new_i] += row[h+ 24]
            answers2[new_i] -= row[h]
            new_i += 1
    answers = answers2



    # model_translator = model_translator()


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

    correct_best = total_best = div_best = 0
    input = X_test
    answers = y_test
    isup = 0
    jsup = 0
    correctsup = 0
    max = np.max(input) * 2
    min = np.min(input) * 2
    step = (max - min) / 10
    best = 0
    for i in range (12):
        devider = 1.5 + (i/12)
        per_comp = 0.5 #number commes frome experiments see code in comment above
        sum_comp = (min + max )/devider
        new_predictions = np.ones((input.shape[0]))
        for row in range(len(input)):
            if input[row][0] + input[row][1] >= sum_comp and (input[row][0]/(input[row][0] + input[row][1])) >= per_comp:
                new_predictions[row] = 0
            elif input[row][0] + input[row][1] >= sum_comp and (input[row][1]/(input[row][0] + input[row][1])) >= per_comp:
                new_predictions[row] = 2

        correct = [0, 0, 0]
        total = [0, 0, 0]
        rong = [0, 0, 0]
        for row in range(len(new_predictions)):
            if new_predictions[row] == answers[row]:
                correct[int(answers[row])] += 1
            elif answers[row] == 1:
                rong[1] += 1
            elif new_predictions[row] == 0 and answers[2]:
                rong[0] += 1
            elif new_predictions[row] == 2 and answers[0]:
                rong[2] += 1
            total[int(answers[row])] += 1

        div = correct[0] - rong[0] + correct[2] - rong[2]
        if div > best:
            best = div
            correct_best = correct
            total_best = total
            div_best = div
            best_rong = rong

    result = 100*(1 + div_best / (correct_best[0] + correct_best[2]))
    if result > best_result:
        best_result = result
        best_model = model


best_model.save('my_model.keras')
print(result)
