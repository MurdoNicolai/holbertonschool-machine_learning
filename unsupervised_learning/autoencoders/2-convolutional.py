#!/usr/bin/env python3
""" contains autoencoders"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
        creates an autoencoder:
        input_dims -> int containing the dimensions of the model input
        hidden_layers -> list with the number of nodes for each hidden layer

        latent_dims -> int containing the dimensions of the latent space
    """
    e_Input = keras.layers.Input(shape=input_dims)
    Output = e_Input
    for nb_filters in filters:
        Output = keras.layers.Conv2D(nb_filters, (3, 3), activation='relu',
                                     padding='same')(Output)
        Output = keras.layers .MaxPooling2D((2, 2), padding='same')(Output)

    encoder = keras.Model(inputs=e_Input, outputs=Output, name='encoder')

    d_Input = keras.layers.Input(latent_dims,)
    Output = d_Input
    for nb_filters in filters[::-2]:
        Output = keras.layers .Conv2D(nb_filters, (3, 3), activation='relu',
                                      padding='same')(Output)
        Output = keras.layers .UpSampling2D((2, 2))(Output)

    Output = keras.layers .Conv2D(filters[-1], (3, 3), activation='relu',
                                  padding='valid')(Output)
    Output = keras.layers .UpSampling2D((2, 2))(Output)

    Output = keras.layers.Dense(input_dims[2], activation='sigmoid')(Output)
    decoder = keras.Model(inputs=d_Input, outputs=Output, name='decoder')

    auto_out = decoder(encoder(e_Input))
    auto = keras.Model(inputs=e_Input, outputs=auto_out, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
