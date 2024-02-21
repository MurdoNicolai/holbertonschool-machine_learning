#!/usr/bin/env python3
""" contains models for machine translation"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def filter(x, y, max_len = 50):
    if len(x) < max_len and len(y) < max_len:
        return True
    return False

def pad_sequences(x, y, max_len=50):
    x = tf.pad(x, paddings=[[0, max_len - tf.size(x)]])
    y = tf.pad(y, paddings=[[0, max_len - tf.size(y)]])
    return x, y

def create_masks(inputs, target):
    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder target padding mask
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Combined mask for the 1st attention block in the decoder
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((tf.shape(target)[1],
                                                       tf.shape(target)[1])), -1, 0)
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)
    combined_mask = combined_mask[:, tf.newaxis, :, :]

    # Decoder padding mask
    decoder_mask = target_padding_mask

    return encoder_mask, combined_mask, decoder_mask
