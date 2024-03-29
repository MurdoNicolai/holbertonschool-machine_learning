#!/usr/bin/env python3
""" contains modules for attention algorythms"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """conatins the class for attention algorythms"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """initialize the class"""
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """calls class"""
        # Multi-Head Attention 1 (masked self-attention)
        attention_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        out1 = self.layernorm1(x + attention_output1)

        # Multi-Head Attention 2 (encoder-decoder attention)
        attention_output2, _ = self.mha2(out1, encoder_output, encoder_output,
                                         padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        out2 = self.layernorm2(out1 + attention_output2)

        # Feed Forward layer
        dense_hidden_output = self.dense_hidden(out2)
        dense_output = self.dense_output(dense_hidden_output)
        dense_output = self.dropout3(dense_output, training=training)
        out3 = self.layernorm3(out2 + dense_output)

        return out3
