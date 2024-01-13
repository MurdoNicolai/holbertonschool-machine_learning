#!/usr/bin/env python3
""" contains modules for attention algorythms"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        # x: previous word index (batch, 1)
        # s_prev: previous hidden state (batch, units)
        # hidden_states: encoder outputs (batch, input_seq_len, units)

        # Embedding layer
        x = self.embedding(x)  # (batch, 1, embedding)

        # Concatenate context vector with x
        context, attention_weights = self.attention(s_prev, hidden_states)
        context = tf.expand_dims(context, 1)  # (batch, 1, units)
        x = tf.concat([context, x], axis=-1)  # (batch, 1, units + embedding)

        # GRU layer
        outputs, s = self.gru(x, initial_state=s_prev)

        # Fully connected layer
        y = self.F(outputs)  # (batch, 1, vocab)
        y = tf.squeeze(y, axis=1)  # (batch, vocab)

        return y, s
