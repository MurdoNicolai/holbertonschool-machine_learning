#!/usr/bin/env python3
import tensorflow as tf
""" contains modules for attention algorythms"""


class RNNEncoder(tf.keras.layers.Layer):
    """ encode for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Sets the following public instance attributes:
        batch - the batch size
        units - the number of hidden units in the RNN cell
        embedding - a keras Embedding layer that converts words
                    from the vocabulary into an embedding vector
        gru - a keras GRU layer with units units
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """
        initializes the hidden states for the RNN cell to a tensor of zeros
        """
        RNN_cell = tf.zeros(shape=(self.batch, self.units))
        return RNN_cell

    def call(self, x, initial):
        """
        calls the encoder
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
