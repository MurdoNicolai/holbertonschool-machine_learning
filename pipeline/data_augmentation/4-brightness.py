#!/usr/bin/env python3
import tensorflow as tf

def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
    - image: a 3D tf.Tensor containing the image to change
    - max_delta: the maximum amount the image should be brightened (or darkened)

    Returns:
    - The altered image
    """
    return tf.image.random_brightness(image, max_delta)
