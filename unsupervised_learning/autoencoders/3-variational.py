#!/usr/bin/env python3
"""Contains variational autoencoders"""
import tensorflow as tf
import tensorflow.keras as keras


def sampling(args):
    """Sample from the distribution"""
    mean, log_var = args
    batch = tf.shape(mean)[0]
    dim = tf.shape(mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return mean + tf.exp(0.5 * log_var) * epsilon

def variational_autoencoder(input_dims, filters, latent_dims, lambtha):
    """
    Creates a variational autoencoder.

    Parameters:
        keras.layers.input_dims (tuple): Dimensions of the keras.layers.input
        filters (list): List with the number of filters for each convolutional layer.
        latent_dims (int): Dimensions of the latent space.
        lambtha (float): Regularization parameter.

    Returns:
        tuple: Encoder, Decoder, VAE models.
    """
    p = lambtha

    # Encoder
    e_Input = keras.layers.Input(shape=input_dims)
    x = e_Input
    for nb_filters in filters:
        x = keras.layers.Conv2D(nb_filters, (3, 3), activation='relu',
                                padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = keras.layers.Flatten()(x)
    mean = keras.layers.Dense(latent_dims)(x)
    log_var = keras.layers.Dense(latent_dims)(x)

    # Use keras.layers.Lambda layer for the sampling function
    latent = keras.layers.Lambda(sampling,
                                 output_shape=(latent_dims,))([mean, log_var])

    encoder = tf.keras.Model(inputs=e_Input,
                             outputs=[mean, log_var, latent], name='encoder')

    # Decoder
    d_Input = keras.layers.Input(shape=(latent_dims,))
    x = keras.layers.Dense(filters[-1] *
                           (input_dims[0] // 2 ** len(filters)) *
                           (input_dims[1] // 2 ** len(filters)),
                           activation='relu')(d_Input)
    x = keras.layers.Reshape((input_dims[0] // 2 ** len(filters),
                              keras.layers.input_dims[1] // 2 ** len(filters),
                              filters[-1]))(x)

    for nb_filters in filters[::-1]:
        x = keras.layers.Conv2DTranspose(nb_filters, (3, 3),
                                         activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)

    decoded = keras.layers.Conv2DTranspose(input_dims[2], (3, 3),
                                           activation='sigmoid',
                                           padding='same')(x)

    decoder = tf.keras.Model(inputs=d_Input, outputs=decoded, name='decoder')

    # VAE
    auto_input = keras.layers.Input(shape=input_dims)
    mean, log_var, latent = encoder(auto_input)
    auto_out = decoder(latent)

    auto = tf.keras.Model(inputs=auto_input, outputs=auto_out,
                          name='variational_autoencoder')

    # Add KL divergence as a regularization term
    kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) -
                                   tf.exp(log_var), axis=-1)
    vae_loss = 'binary_crossentropy' + (lambtha * tf.reduce_mean(kl_loss))

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
