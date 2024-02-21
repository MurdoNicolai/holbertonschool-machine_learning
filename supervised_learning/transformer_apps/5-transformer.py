import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dropout, Dense
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm

        self.depth = dm // h

        self.Wq = Dense(dm)
        self.Wk = Dense(dm)
        self.Wv = Dense(dm)

        self.linear = Dense(dm)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, Q, K, V, mask):
        matmul_qk = tf.matmul(Q, K, transpose_b=True)

        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, V)

        return output, attention_weights

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, dm):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.positional_encoding(max_seq_len, dm)

    def get_angles(self, pos, i, dm):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(dm, tf.float32))
        return pos * angle_rates

    def positional_encoding(self, max_seq_len, dm):
        angle_rads = self.get_angles(
            pos=tf.range(max_seq_len, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(dm, dtype=tf.float32)[tf.newaxis, :],
            dm=dm
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = tf.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = tf.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.encoding[:, :tf.shape(x)[1], :]

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = Dense(hidden, activation='relu')
        self.dense_output = Dense(dm)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(drop_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        hidden_output = self.dense_hidden(out1)
        output = self.dense_output(hidden_output)

        return output

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = Dense(hidden, activation='relu')
        self.dense_output = Dense(dm)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(drop_rate)
        self.dropout2 = Dropout(drop_rate)
        self.dropout3 = Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        hidden_output = self.dense_hidden(out2)
        output = self.dense_output(hidden_output)

        output = self.dropout3(output, training=training)
        output = self.layernorm3(output + out2)

        return output

class Transformer(tf.keras.Model):
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)

        self.final_layer = Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training, look_ahead_mask, decoder_mask)

        final_output = self.final_layer(dec_output)

        return final_output
