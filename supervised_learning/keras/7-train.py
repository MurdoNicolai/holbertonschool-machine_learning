#!/usr/bin/env python3
""" has optimize_model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model with optional validation, early stopping, and learning
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

        if learning_rate_decay:
            def lr_schedule(epoch, lr):
                return alpha / (1 + decay_rate * epoch)

            lr_decay_callback = K.callbacks.LearningRateScheduler(
                lr_schedule,
                verbose=1
            )
            callbacks.append(lr_decay_callback)

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data, callbacks=callbacks)

    return history
