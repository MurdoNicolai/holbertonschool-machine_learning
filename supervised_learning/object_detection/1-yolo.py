#!/usr/bin/env python3
"""contains the yolo class"""
import tensorflow as tf
import numpy as np


def softmax(x):
    """returns the softmax of x"""
    return np.exp(x) / np.sum(np.exp(x), axis=3, keepdims=True)


def sigmoid(x):
    """returns the sigmoid of x"""
    return 1 / (1 + np.exp(-x))


def process_outputs(self, outputs, image_size):
    boxes = []
    box_confidences = []
    box_class_probs = []

    input_height, input_width = image_size
    anchorcount = 0

    for output in outputs:
        grid_height, grid_width, num_anchors, _ = output.shape

        # Applying sigmoid activation to the confidence score
        output[..., 4] = 1.0 / (1.0 + np.exp(-output[..., 4]))

        # Split the output into parts
        center_x, center_y, width, height, box_confidence, class_probs = np.split(output, (1, 2, 3, 4, 5), axis=-1)

        # Applying sigmoid activation to center_x and center_y
        center_x = 1.0 / (1.0 + np.exp(-center_x))
        center_y = 1.0 / (1.0 + np.exp(-center_y))

        # Exponentiate to get width and height
        width = np.exp(width)
        height = np.exp(height)

        # Adjust coordinates to the image size
        center_x = (center_x + np.arange(grid_width)) / grid_width * input_width
        center_y = (center_y + np.arange(grid_height)) / grid_height * input_height
        width = width * self.anchors[anchorcount, 0]
        height = height * self.anchors[anchorcount, 1]

        # Calculate bounding box coordinates
        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0

        # Append the results for this anchor box to the respective lists
        boxes.append(np.concatenate((x1, y1, x2, y2), axis=-1))
        box_confidences.append(box_confidence)
        box_class_probs.append(softmax(class_probs))

        anchorcount += 1

    return boxes, box_confidences, box_class_probs
