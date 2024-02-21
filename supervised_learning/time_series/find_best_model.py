#!/usr/bin/env python3
"""contains tenserflow stuff"""

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.initializers import GlorotUniform
from preprocess_data import read_csv, read_csv_api
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

class EntropyRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, rate=0.001):
        super(EntropyRegularizer, self).__init__()
        self.rate = rate

    def __call__(self, y_true, y_pred):
        # Compute the entropy of the predictions
        entropy = tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10))

        # Apply the regularization penalty
        penalty = -self.rate * entropy

        return penalty

def model(dropout_rate, regularizer):
    model = Sequential()

    # Add the first SimpleRNN layer with input shape (2, 24)
    model.add(SimpleRNN(units=120, activation='relu', input_shape=(2, 24),
                        kernel_initializer=GlorotUniform(), return_sequences=True))
    model.add(Dropout(dropout_rate))

    # Add the second SimpleRNN layer with the same number of units
    # and input_shape matching the output shape of the previous layer
    model.add(SimpleRNN(units=120, activation='relu',
                        kernel_initializer=GlorotUniform()))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())

    # Continue with the rest of the model
    model.add(Dense(units=192, activation='relu', kernel_regularizer=l2(regularizer)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=48, activation='sigmoid'))

    def loss_with_entropy_regularization(y_true, y_pred):
        # Compute the binary crossentropy loss
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Add the entropy regularization penalty
        entropy = loss - tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10))
        penalty = EntropyRegularizer(rate=0.01)(y_true, y_pred)

        return loss + penalty

    # Compile the model with the custom loss function
    model.compile(loss=loss_with_entropy_regularization, optimizer='adam')

    return model
drop = 0.5
reg = 0.01
model = model(drop, reg)


data = read_csv_api() #change to read_csv for more (non api) data
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
# Train the model
epochs = 10
batch_size = 32
best_result = 0
best_model = None

for i in range (100):
    early_stopping = EarlyStopping(patience=2)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
    predictions = model.predict(training)
    # reorganise data by hour ignoring days
    training2 = np.zeros((predictions.shape[0]*24, 2))
    new_i = 0
    for row in predictions:
        for h in range(24):
            training2[new_i][0] = row[h]
            training2[new_i][1] = row[h+ 24]
            new_i += 1

    answers2 = np.ones((answers.shape[0]*24))
    new_i = 0
    for row in answers:
        for h in range(24):
            answers2[new_i] += row[h+ 24]
            answers2[new_i] -= row[h]
            new_i += 1



    # model_translator = model_translator()


    # Split the data into training2 and validation sets
    validation_split = 0.1
    test_split = 0.1
    validation_samples = int(training2.shape[0] * validation_split)
    test_samples = int(training2.shape[0] * test_split)


    X_train1 = training2[:-(validation_samples + test_samples)]
    y_train1 = answers2[:-(validation_samples + test_samples)]
    X_val1 = training2[-(validation_samples + test_samples):-test_samples]
    y_val1 = answers2[-(validation_samples + test_samples):-test_samples]
    X_test1 = training2[-test_samples:]
    y_test1 = answers2[-test_samples:]

    input = X_train1
    answers2 = y_train1
    isup = 0
    jsup = 0
    correctsup = 0
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
    rong = [0, 0, 0]
    for row in range(len(new_predictions)):
        if new_predictions[row] == answers2[row]:
            correct[int(answers2[row])] += 1
        elif answers2[row] == 1:
            rong[1] += 1
        elif new_predictions[row] == 0 and answers2[2]:
            rong[0] += 1
        elif new_predictions[row] == 2 and answers2[0]:
            rong[2] += 1
        total[int(answers2[row])] += 1

    print(correct, rong)
    div = correct[0] - rong[0] + correct[2] - rong[2]
    result = 100*(1 + div / (correct[0] + correct[2]))
    if result > best_result:
        best_result = result
        best_model = model


best_model.save('my_model_api3.keras')
print(best_result)
