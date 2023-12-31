#!/usr/bin/env python3
""" has optimize_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with optional validation
    """
    callbacks = []

    if validation_data is not None:
        val_data, val_labels = validation_data
        validation_data = (val_data, val_labels)

        if early_stopping:
            early_stopping_callback = K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                verbose=verbose,
                restore_best_weights=True
            )
            callbacks.append(early_stopping_callback)

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data, callbacks=callbacks)

    return history
