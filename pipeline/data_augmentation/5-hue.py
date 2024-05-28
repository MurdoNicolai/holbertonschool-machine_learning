#!/usr/bin/env python3
import tensorflow as tf

def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
    - image: a 3D tf.Tensor containing the image to change
    - delta: the amount the hue should change

    Returns:
    - The altered image
    """
    return tf.image.adjust_hue(image, delta)
