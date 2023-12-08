#!/usr/bin/env python3
""" contains autoencoders"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
        creates an autoencoder:
        input_dims -> int containing the dimensions of the model input
        hidden_layers -> list with the number of nodes for each hidden layer

        latent_dims -> int containing the dimensions of the latent space
    """
    e_Input = keras.layers.Input(input_dims,)
    Output = e_Input
    for nb_nodes in hidden_layers:
        Output = keras.layers.Dense(nb_nodes, activation='relu')(Output)
    Output = keras.layers.Dense(latent_dims, activation='relu',
    activity_regularizer=keras.regularizers.L1(lambtha))(Output)
    encoder = keras.Model(inputs=e_Input, outputs=Output, name='encoder')

    d_Input = keras.layers.Input(latent_dims,)
    Output = d_Input
    for nb_nodes in hidden_layers[::-1]:
        Output = keras.layers.Dense(nb_nodes, activation='relu')(Output)
    Output = keras.layers.Dense(input_dims, activation='sigmoid')(Output)
    decoder = keras.Model(inputs=d_Input, outputs=Output, name='decoder')

    auto_input = keras.layers.Input(shape=(input_dims,))
    auto_out = decoder(encoder(auto_input))

    auto = keras.Model(inputs=auto_input, outputs=auto_out, name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
