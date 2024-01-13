#!/usr/bin/env python3
""" contains modules for attention algorythms"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Scaled Dot Product Attention
    Args:
    - Q: Query matrix with shape (..., seq_len_q, dk)
    - K: Key matrix with shape (..., seq_len_v, dk)
    - V: Value matrix with shape (..., seq_len_v, dv)
    - mask: Optional mask with shape (..., seq_len_q, seq_len_v)

    Returns:
    - output: Scaled Dot Product Attention output with shape
    - weights: Attention weights with shape (..., seq_len_q, seq_len_v)
    """

    # Get the depth of the key matrix
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # Calculate the scaled dot product attention scores
    scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk)

    # Apply the mask if provided
    if mask is not None:
        scores += (mask * -1e9)

    # Apply the softmax activation to get attention weights
    weights = tf.nn.softmax(scores, axis=-1)

    # Multiply the attention weights with the value matrix
    output = tf.matmul(weights, V)

    return output, weights
