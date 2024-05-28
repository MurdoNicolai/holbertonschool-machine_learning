#!/usr/bin/env python3
import tensorflow as tf

def flip_image(image):
    """
    Flips an image horizontally.

    Args:
    - image: a 3D tf.Tensor containing the image to flip

    Returns:
    - The flipped image
    """
    return tf.image.flip_left_right(image)
