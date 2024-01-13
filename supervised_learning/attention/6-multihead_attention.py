#!/usr/bin/env python3
""" contains modules for attention algorythms"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """conatins the class for attention algorythms"""

    def __init__(self, dm, h):
        """initializes"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """calls class"""
        batch_size = tf.shape(Q)[0]

        # Apply linear transformations to obtain Q, K, and V for each head
        Q = self.Wq(Q)  # (batch, seq_len_q, dm)
        K = self.Wk(K)  # (batch, seq_len_v, dm)
        V = self.Wv(V)  # (batch, seq_len_v, dm)

        # Split Q, K, and V into multiple heads
        Q = tf.reshape(Q, (batch_size, -1, self.h, self.depth))
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])  # (batch, h, seq_len_q, depth)

        K = tf.reshape(K, (batch_size, -1, self.h, self.depth))
        K = tf.transpose(K, perm=[0, 2, 1, 3])  # (batch, h, seq_len_v, depth)

        V = tf.reshape(V, (batch_size, -1, self.h, self.depth))
        V = tf.transpose(V, perm=[0, 2, 1, 3])  # (batch, h, seq_len_v, depth)

        # Apply scaled dot product attention to each head
        attention_output, attention_weights = sdp_attention(Q, K, V, mask)

        # Concatenate the heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1,
                                                         self.dm))
        # Apply linear transformation to obtain the final output
        output = self.linear(attention_output)
        return output, attention_weights
