#!/usr/bin/env python3
""" contains modules for attention algorythms"""
import tensorflow as tf


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

class SelfAttention(tf.keras.layers.Layer):
    """ calculate the attention for machine translation """

    def __init__(self, units):
        """Sets the following public instance attributes:

            W - a Dense layer with units units
            U - a Dense layer with units units
            V - a Dense layer with 1 units
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        # Expand dimensions of s_prev to (batch, 1, units) for broadcasting
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Apply the Dense layers to calculate W, U, and V
        W_s = self.W(s_prev_expanded) 
        U_hs = self.U(hidden_states)
        tanh_input = tf.tanh(W_s + U_hs)

        # Apply the final Dense layer V to get attention scores
        attention_scores = self.V(tanh_input)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        # Calculate context vector by taking the weighted sum of hidden states
        context = tf.reduce_sum(attention_weights * hidden_states, axis=1)

        return context, attention_weights



