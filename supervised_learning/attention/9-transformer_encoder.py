#!/usr/bin/env python3
""" contains modules for attention algorythms"""
import tensorflow as tf
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock
positional_encoding = __import__('4-positional_encoding').positional_encoding


class Encoder(tf.keras.layers.Layer):
    """conatins the class for attention algorythms"""
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in
                       range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """calls class"""
        seq_len = tf.shape(x)[1]

        # Embedding and positional encoding
        x = self.embedding(x)  # (batch, input_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        # Apply dropout to the positional encodings
        x = self.dropout(x, training=training)

        # Pass the input through each encoder block
        for block in self.blocks:
            x = block(x, training, mask)

        return x
