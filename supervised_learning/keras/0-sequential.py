import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""

    # Create a Sequential model
    model = K.Sequential()

    # Iterate through the layers and activations
    for layer_size, activation in zip(layers, activations):
        # Add a Dense layer with L2 regularization
        model.add(K.layers.Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=K.regularizers.l2(lambtha),
            input_shape=(nx,) if not model.layers else ()
        ))

        # Add Dropout layer (if keep_prob < 1.0)
        if keep_prob < 1.0:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
