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


class Yolo():
    """uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r") as myfile:
            self.class_names = myfile.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Returns a tuple of (boxes, box_confidences, box_class_probs):
        boxes -> contains the processed boundary boxes
        box_confidences -> contains the box confidences for each output
        box_class_probs -> contains the box class probabilities for each output
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        input_height = image_size[0]
        input_width = image_size[1]
        anchorcount = 0
        for output in outputs:
            output[..., 4] = 1.0 / (1.0 + np.exp(-output[..., 4]))
            splt_out = np.array_split(output, (4, 5, ), axis=3)
            grid_height, grid_width, num_anchors, _ = output.shape

            splt_out[0][..., :2] = 1.0 / (1.0 + np.exp(-splt_out[0][..., :2]))
            splt_out[0][..., 2:4] = np.exp(splt_out[0][..., 2:4])
            centerx = splt_out[0][..., :1] * input_width / grid_width
            centery = splt_out[0][..., 1:2] * input_height / grid_height
            anchorx = self.anchors[anchorcount][..., 0]
            anchorx = anchorx.reshape(1, 1, num_anchors, 1)
            anchory = self.anchors[anchorcount][..., 1]
            anchory = anchory.reshape(1, 1, num_anchors, 1)
            widthx = (splt_out[0][..., 2:3] * anchorx
                      * input_width / grid_width / 15)
            widthy = (splt_out[0][..., 3:4] * anchory
                      * input_height / grid_height / 15)

            splt_out[0][..., :1] = centerx - widthx
            splt_out[0][..., 1:2] = centery - widthy
            splt_out[0][..., 2:3] = centerx + widthx
            splt_out[0][..., 3:4] = centery + widthy

            boxes.append(splt_out[0])
            box_confidences.append(splt_out[1])
            box_class_probs.append(sigmoid(splt_out[2]))
            anchorcount += 1
        return ((boxes, box_confidences, box_class_probs))
