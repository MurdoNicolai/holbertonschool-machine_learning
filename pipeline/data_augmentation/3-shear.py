#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_addons as tfa

def shear_image(image, intensity):
    """
    Randomly shears an image.

    Args:
    - image: a 3D tf.Tensor containing the image to shear
    - intensity: the intensity with which the image should be sheared

    Returns:
    - The sheared image
    """
    return tfa.image.shear_x(image, intensity, replace=0)
