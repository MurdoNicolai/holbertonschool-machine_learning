#!/usr/bin/env python3
"""Contains autoencoders"""
import tensorflow.keras as keras

def autoencoder(input_shape, filters, latent_dims, lambtha):
    """
    Creates a convolutional autoencoder.

    Parameters:
        input_shape (tuple): Dimensions of the input data
        filters (list): List with the number of filters
        latent_dims (int): Dimensions of the latent space.
        lambtha (float): Regularization parameter.

    Returns:
        tuple: Encoder, Decoder, Autoencoder models.
    """
    p = lambtha

    # Encoder
    e_Input = keras.layers.Input(shape=input_shape)
    x = e_Input
    for num_filters in filters:
        x = keras.layers.Conv2D(num_filters, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = keras.layers.Flatten()(x)
    encoded = keras.layers.Dense(latent_dims, activation='relu',
                                 activity_regularizer=keras.regularizers.L1(p))(x)

    encoder = keras.Model(inputs=e_Input, outputs=encoded, name='encoder')

    # Decoder
    d_Input = keras.layers.Input(shape=(latent_dims,))
    x = keras.layers.Dense(filters[-1] * (input_shape[0] // 2 **
                                          len(filters)) *
                           (input_shape[1] // 2 ** len(filters)),
                           activation='relu')(d_Input)
    x = keras.layers.Reshape((input_shape[0] // 2 ** len(filters),
                              input_shape[1] // 2 ** len(filters),
                              filters[-1]))(x)

    for num_filters in filters[::-1]:
        x = keras.layers.Conv2DTranspose(num_filters, (3, 3),
                                         activation='relu',
                                         padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    decoded = keras.layers.Conv2DTranspose(input_shape[2], (3, 3),
                                           activation='sigmoid',
                                           padding='same')(x)

    decoder = keras.Model(inputs=d_Input, outputs=decoded, name='decoder')

    # Autoencoder
    auto_input = keras.layers.Input(shape=input_shape)
    auto_out = decoder(encoder(auto_input))

    auto = keras.Model(inputs=auto_input, outputs=auto_out, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
