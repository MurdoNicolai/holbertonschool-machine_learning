#!/usr/bin/env python3
"""contains the yolo class"""
import tensorflow as tf


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
