#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Lambda, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def preprocess_data(X, Y):
    """
    Pre-process the data for the model.

    Args:
    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data.
    Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X.

    Returns:
    X_p: numpy.ndarray containing the preprocessed X.
    Y_p: numpy.ndarray containing the preprocessed Y.
    """
    X_p = X.astype('float32') / 255.0
    Y_p = to_categorical(Y, 10)
    return X_p, Y_p

def build_model(input_shape):
    """
    Builds the convolutional neural network model using a pre-trained application.

    Args:
    input_shape: tuple, the shape of the input data.

    Returns:
    model: the compiled Keras model.
    """
    # Load the pre-trained VGG16 model without the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create the input layer and resize images
    inputs = Input(shape=input_shape)
    resize = Lambda(lambda image: tf.image.resize(image, (224, 224)))(inputs)

    # Get the base model output
    base_output = base_model(resize, training=False)

    # Add custom layers on top of the base model
    x = Flatten()(base_output)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def main():
    # Load the CIFAR-10 data
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Preprocess the data
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Build the model
    model = build_model(X_train_p.shape[1:])

    # Train the model
    model.fit(X_train_p, Y_train_p, validation_data=(X_test_p, Y_test_p), epochs=10, batch_size=64)

    # Save the model
    model.save('cifar10.h5')

if __name__ == '__main__':
    main()
